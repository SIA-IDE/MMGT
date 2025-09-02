import concurrent.futures
import os
import random
from pathlib import Path
import torch
from PIL import Image
import numpy as np

from src.dwpose import DWposeDetector
from src.dwpose import DWposeDetector_movment_mask
from src.utils.util import get_fps, read_frames, save_videos_from_pil

from pathlib import Path
import shutil

def reshape_keypoints_to_list(np_keypoints_info):
    seq_len = np_keypoints_info.shape[0] # 60
    # np_keypoints_info = random_mask(np_keypoints_info)
    keypoints_list = [np_keypoints_info[i:i+1] for i in range(seq_len)]
    return keypoints_list


def random_mask(keypoints_sequence):

    seq_len = keypoints_sequence.shape[0]        
    keypoints_sequence = keypoints_sequence.reshape(seq_len, 134, 3)
    # 定义身体部位的索引范围
    body_parts = {
        'lips': range(72, 92),
        'torso': range(0, 8),
        'head': range(14, 60),
        'hands': list(range(92, 113)) + list(range(113, 134)), # 左右手分别占用不同区域
        'eyes': list(range(60, 66)) + list(range(66, 72))  # 左右眼分别占用不同区域
    }
    
    # 始终掩码的部位
    fixed_parts = {
        'left_leg': range(9, 11),
        'right_leg': range(12, 14)
    }

    # 设置掩码比例（1-5个部位的比例）
    mask_distribution = [0.6, 0.3, 0.05, 0.025, 0.025]
    num_parts_to_mask = np.random.choice([1, 2, 3, 4, 5], p=mask_distribution)

    # 随机选择要掩码的部位
    selected_parts = random.sample(list(body_parts.keys()), num_parts_to_mask)

    # 创建掩码矩阵
    mask = np.ones(keypoints_sequence.shape, dtype=np.uint8)

    # 对固定部位（左腿和右腿）应用掩码
    for part, indices in fixed_parts.items():
        mask[:, indices, :] = 0  # 针对整个视频序列掩码

    # 对随机选中的其他部位应用掩码
    for part in selected_parts:
        indices = body_parts[part]
        mask[:, indices, :] = 0  # 针对整个视频序列掩码

    # 应用掩码
    masked_keypoints_sequence = keypoints_sequence * mask
    # selected_parts + list(fixed_parts.keys())
    return masked_keypoints_sequence


def mask_leg(normalized_keypoints):
    seq_len = normalized_keypoints.shape[0]        
    keypoints = normalized_keypoints.reshape(seq_len, 134, 3)
    mask = np.ones(keypoints.shape, dtype=np.uint8) 

    left_leg = range(9, 11)       # 左腿
    right_leg = range(12,14)      # 右腿

    torso = range(0, 18)  # 躯干
    left_hands = range(92, 113)  # 左手
    right_hands = range(113, 134) # 右手
    all_face = range(14, 60)  # 整张脸
    lip_indices = range(72, 92) # 嘴唇
    left_eye_indices = range(60, 66)  # 左眼
    right_eye_indices = range(66, 72) # 右眼  
    chin_indices = range(28, 37)      # 下巴

    mask[:, left_leg, :] = 0
    mask[:, right_leg, :] = 0

    # mask[:, lip_indices, :] = 0
    # mask[:, torso, :] = 0  # 躯干
    # mask[:, hands, :] = 0
    # mask[:, left_eye_indices, :] = 0
    # mask[:, right_eye_indices, :] = 0
    # mask[:, lip_indices, :] = 0
    normalized_keypoints = keypoints * mask
    keypoints = normalized_keypoints.reshape(seq_len, -1)
    return keypoints


def process_keypoints(np_keypoints): # np_keypoints.shape=(60, 402)
    detector_data = DWposeDetector_movment_mask()    
    keypoints_list = reshape_keypoints_to_list(np_keypoints)
    kps_results = []
    kps_results_hands = []
    kps_results_lips = []
    kps_results_faces = []
    for keypoints_info in keypoints_list: # keypoints_info.shape=(1, 402)
        # 3.reshape
        keypoints_info = mask_leg(keypoints_info)
        seq_len = keypoints_info.shape[0]   
        # keypoints_info = random_mask(keypoints_info)
        keypoints_info = keypoints_info.reshape(seq_len, 134, 3)
        keypoints, scores = keypoints_info[..., :2], keypoints_info[..., 2]
        kps_result, kps_results_hand, kps_results_lip, kps_results_face, score = detector_data(keypoints, scores)

        kps_results.append(kps_result)
        kps_results_lips.append(kps_results_lip)
        kps_results_hands.append(kps_results_hand)
        kps_results_faces.append(kps_results_face)
    # 第一个是正常pose视频，第二个是嘴唇掩码，第三个是面部和手部掩码
    return kps_results, kps_results_hands, kps_results_lips, kps_results_faces, score

def normalize(data):
    min_val = -200 
    max_val = 800
    normalized_data = (data - min_val) / (max_val - min_val)
    normalized_data = normalized_data * 2 - 1
    return normalized_data

def denormalize(data):
    min_val = -200 
    max_val = 800
    data = (data + 1) / 2
    return data * (max_val - min_val) + min_val


# def normalize(data):
#     if isinstance(data, np.ndarray):
#         data = torch.from_numpy(data)    
#     normalized_data = torch.sigmoid(data)
#     normalized_data = normalized_data * 2 - 1 # 将数据映射到[-1, 1]
#     return normalized_data.numpy()

# def denormalize(data):
#     if isinstance(data, np.ndarray):
#         data = torch.from_numpy(data)  
#     fg_kp = (data + 1) / 2
#     x_reconstructed = - torch.log(1 / fg_kp - 1)
#     return x_reconstructed.numpy()



def process_single_video(video_path, detector, root_dir, save_dir, video2video=bool):
    relative_path = os.path.relpath(video_path, root_dir)
    print(relative_path, video_path, root_dir)
    # out_path = os.path.join(save_dir, relative_path)
    out_path_dwpose = os.path.join(save_dir, "dwpose", os.path.splitext(relative_path)[0] + ".mp4")
    os.makedirs(os.path.dirname(out_path_dwpose), exist_ok=True)
    out_path_face = os.path.join(save_dir, "face", os.path.splitext(relative_path)[0] + ".mp4")
    os.makedirs(os.path.dirname(out_path_face), exist_ok=True)
    out_path_lips = os.path.join(save_dir, "lips", os.path.splitext(relative_path)[0] + ".mp4")
    os.makedirs(os.path.dirname(out_path_lips), exist_ok=True)
    out_trash_path = os.path.join(save_dir, "trash", os.path.splitext(relative_path)[0] + ".mp4")
    os.makedirs(os.path.dirname(out_trash_path), exist_ok=True)

    if os.path.exists(out_path_dwpose):
        print("视频已处理")
        return

    output_dir = Path(os.path.dirname(os.path.join(save_dir, relative_path)))
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if video2video == True :    
        fps = get_fps(video_path)

        if fps is None:
            try:
                target = out_trash_path / video_path.name
                # 若重名则在文件名前追加时间戳，避免覆盖
                if target.exists():
                    target = target.with_stem(f"{target.stem}_{int(time.time())}")
                shutil.move(str(video_path), str(target))
                print(f"[INFO] 已将损坏视频 {video_path} 移动到 {target}")
            except Exception as e:
                # 如果移动失败，只记录警告，不中断流程
                print(f"[WARN] 移动 {video_path} 到损坏目录失败: {e}")
            return  # 跳过后续处理



        frames = read_frames(video_path)

        # 2.合并
        kps_results = []
        keypoints_list = []
        for i, frame_pil in enumerate(frames):
            _, _, whole_keypoints = detector(frame_pil)
            # score = np.mean(score, axis=-1)
            if whole_keypoints is not None:
                if whole_keypoints.shape[0] != 1:
                    whole_keypoints = whole_keypoints = whole_keypoints[:1, :]
                whole_keypoints = whole_keypoints.copy()        
            whole_keypoints = whole_keypoints.squeeze(0)
            keypoints_list.append(whole_keypoints)


        # 合并完成
        keypoints = np.array(keypoints_list) # keypoints.shape=(60, 402)
        normalized_keypoints = normalize(keypoints)

        # 保存为.npy文件(获取npy文件的名称)
        base_name = os.path.basename(video_path)
        base_name_without_ext = os.path.splitext(base_name)[0]   
        save_path = os.path.join(save_dir, "dwpose_npy", f"{base_name_without_ext}.npy") # keypoints.shape=(60, 402)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # TODO 读取.npy文件
        np.save(save_path, normalized_keypoints)

        # 将keypoints合成为视频
        # else:
        # recovered_keypoints = denormalize(normalized_keypoints)
        # kps_results, _, kps_results_lips, kps_results_faces, score = process_keypoints(recovered_keypoints)
        # # 生成没有手部的pose视频
        # save_videos_from_pil(kps_results, out_path_dwpose, fps=fps)
        # # 生成嘴唇部位的掩码
        # save_videos_from_pil(kps_results_lips, out_path_lips, fps=fps)
        # # 生成面部表情和手部的掩码
        # save_videos_from_pil(kps_results_faces, out_path_face, fps=fps)

    else:
        npy2video(video_path, out_path_dwpose, out_path_lips, out_path_face, fps=25)



def npy2video(np_path, out_path_dwpose, out_path_lips, out_path_face, fps=30):

    normalized_keypoints = np.load(np_path)
    # normalized_keypoints = normalized_keypoints * 2 - 1                                                                                               
    recovered_keypoints = denormalize(normalized_keypoints)

    kps_results, _, kps_results_lips, kps_results_faces, score = process_keypoints(recovered_keypoints)

    save_videos_from_pil(kps_results, out_path_dwpose, fps=fps)
    # 生成嘴唇部位的掩码
    save_videos_from_pil(kps_results_lips, out_path_lips, fps=fps)
    # # 生成面部表情和手部的掩码
    save_videos_from_pil(kps_results_faces, out_path_face, fps=fps)

def process_reference_image(reference_path):
    """
    reference_path: 可以是单张图片路径，或只包含这一张 .png 的目录路径
    返回：normalize 后的关键点向量
    """
    # 1) 初始化检测器（按你原来的设定放到 cuda:0 上；没有 CUDA 时自动退回 CPU）
    detector = DWposeDetector()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    detector = detector.to(device)

    # 2) 解析要读的图片路径
    p = Path(reference_path)
    if p.is_dir():
        pngs = sorted(p.glob("*.png"))
        if not pngs:
            raise FileNotFoundError(f"目录 {p} 下没有找到 .png 图片")
        image_path = pngs[0]
    else:
        image_path = p
        if image_path.suffix.lower() != ".png":
            raise ValueError(f"期望传入 .png 图片，但得到的是：{image_path.suffix}")

    # 3) 读取单张图片（保持 RGB）
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # 4) 关键点检测（不需要循环帧了）
    with torch.no_grad():
        _, _, whole_keypoints = detector(img)

    # 5) 基本健壮性处理：确保检测到关键点，只保留第一个人
    # if whole_keypoints is None:
    #     raise RuntimeError(f"在图片 {image_path} 中未检测到人体关键点")
    # if getattr(whole_keypoints, "shape", None) is None or whole_keypoints.shape[0] == 0:
    #     raise RuntimeError(f"在图片 {image_path} 中未检测到人体关键点")

    if whole_keypoints.shape[0] != 1:
        # 只取第一个人的关键点
        whole_keypoints = whole_keypoints[:1, :]

    # 6) 去掉 batch 维度，得到一维向量，如 (402,)
    whole_keypoints = whole_keypoints.squeeze(0)

    # 7) 归一化并返回
    normalized_keypoints = normalize(whole_keypoints)
    return normalized_keypoints


# # 对推理图片提取第一帧的dwpose
# def process_reference_image(reference_image_path):

#     detector = DWposeDetector()
#     # detector_data = DWposeDetector_data()
#     detector = detector.to(f"cuda:0")
#     # 读取图片
#     frames = read_frames(reference_image_path)
#     for i, frame_pil in enumerate(frames):
#     # 处理图片
#         _, _, whole_keypoints = detector(frame_pil)
#         # 如果检测到关键点
#         if whole_keypoints is not None:
#             if whole_keypoints.shape[0] != 1:
#                 whole_keypoints = whole_keypoints = whole_keypoints[:1, :]
#             whole_keypoints = whole_keypoints.copy()        
#         whole_keypoints = whole_keypoints.squeeze(0) # whole_keypoints.shape = (1, 402)
#     normalized_keypoints = normalize(whole_keypoints)
#     return normalized_keypoints


# 对模型生成的结果进行可视化
def pose_vid_generator(normalized_keypoints, out_path_dwpose, out_path_hands, out_path_lips, out_path_face, fps=30):
    recovered_keypoints = denormalize(normalized_keypoints)
    kps_results, kps_results_hands, kps_results_lips, kps_results_faces, score = process_keypoints(recovered_keypoints)
    save_videos_from_pil(kps_results, out_path_dwpose, fps=fps)
    # 生成没有手部的pose视频
    save_videos_from_pil(kps_results_hands, out_path_hands, fps=fps)
    # 生成嘴唇部位的掩码
    save_videos_from_pil(kps_results_lips, out_path_lips, fps=fps)
    # 生成面部表情和手部的掩码
    save_videos_from_pil(kps_results_faces, out_path_face, fps=fps)



def process_batch_videos(video_list, detector, root_dir, save_dir, video2video):
    for i, video_path in enumerate(video_list):
        print(f"Process {i}/{len(video_list)} video")
        process_single_video(video_path, detector, root_dir, save_dir, video2video=video2video)



if __name__ == "__main__":
    import argparse
# 运行该代码的命令行如下，当只提取视频中的kpt数据用于audio2pose模型训练时使用时，输入为视频文件夹，输出为pose_npy文件夹，设置kpt==True
# PYTHONPATH=. python data/extract_movment_mask_all.py --input_root /root/node03-nfs/aaai/DiffTED1/output/val1  --kpt True

# 要想将模型预测的.npy文件重新绘制成pose的时候，运行如下命令行，输入为pose_npy文件夹，输出为pose_video文件夹，设置kpt==False
# CUDA_LAUNCH_BLOCKING=1 python data/extract_movment_mask_all.py --input_root /root/node03-nfs/aaai/hallo/data/sliced_data/train/keypoints --gpu_id 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", default="/root/node03-nfs/aaai/MM-Diffusion/eval/test_video/test/diffgesture_mimicmotion_on_pats/ori", type=str)
    parser.add_argument("--save_dir",  type=str, help="Path to save extracted pose videos")
    parser.add_argument("-j", type=int, default=1, help="Num workers")
    parser.add_argument("--video2video", type=bool, default=True) 
    parser.add_argument("--gpu_id", type=int, default="0")
    # parser.add_argument("-p", "--parallelism", default=8,
    #                     type=int, help="Level of parallelism")
    # parser.add_argument("-r", "--rank", default=0, type=int,
    #                     help="Rank for distributed processing")    

    args = parser.parse_args()
    num_workers = args.j
    if args.save_dir is None:                                                                                                                                                
        save_dir = args.input_root + "_dwpose_lips"
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", f"{args.gpu_id}")
    gpu_ids = [int(id) for id in cuda_visible_devices.split(",")]
    print(f"Available GPU IDs: {gpu_ids}")

    # collect all video paths
    video_mp4_paths = set()
    for root, dirs, files in os.walk(args.input_root):
        for name in files:
            if name.endswith(".mp4"):
                video_mp4_paths.add(os.path.join(root, name))
    video_mp4_paths = list(video_mp4_paths)
    random.shuffle(video_mp4_paths)

    # split into chunks,
    batch_size = (len(video_mp4_paths) + num_workers - 1) // num_workers
    print(f"Num videos: {len(video_mp4_paths)} {batch_size = }")
    video_chunks = [
        video_mp4_paths[i : i + batch_size]
        for i in range(0, len(video_mp4_paths), batch_size)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, chunk in enumerate(video_chunks):
            # init detector
            gpu_id = gpu_ids[i % len(gpu_ids)]
            print(f"Initializing detector on GPU {gpu_id}")
            detector = DWposeDetector()
            detector_data = DWposeDetector_movment_mask()
            torch.cuda.set_device(gpu_id)
            detector = detector.to(f"cuda:{gpu_id}")

            futures.append(
                executor.submit(
                    process_batch_videos, chunk, detector, args.input_root, save_dir, args.video2video
                )
            )
        for future in concurrent.futures.as_completed(futures):
            # try:
            future.result()
            # except Exception as e:
                # print(f"Error occurred: {e}")