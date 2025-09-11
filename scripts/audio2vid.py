import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List
from tqdm import tqdm
import av
import glob
import numpy as np
import torch
import torchvision
import librosa  # 用于加载音频文件
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
import cv2
from src.dataset.audio_processor import AudioProcessor
from src.dataset.image_processor import ImageProcessor
from src.models.audio_proj import AudioProjModel
from scipy.interpolate import CubicSpline
from src.audio2pose_model.SMGA import SMGA
# from args_inference import parse_test_args
from data.slice import slice_audio
from data.audio_extraction.baseline_features import extract as baseline_extract
from data.audio_extraction.wavlm_features import wavlm_init
from data.audio_extraction.wavlm_features import extract_wo_init as wavlm_extract
from data.extract_movment_mask_all import (process_reference_image, pose_vid_generator, mask_leg)
from torch import nn
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames, save_videos_grid

class Net(nn.Module):
    """
    The Net class combines all the necessary modules for the inference process.
    
    Args:
        reference_unet (UNet2DConditionModel): The UNet2DConditionModel used as a reference for inference.
        denoising_unet (UNet3DConditionModel): The UNet3DConditionModel used for denoising the input audio.
        face_locator (FaceLocator): The FaceLocator model used to locate the face in the input image.
        imageproj (nn.Module): The ImageProjector model used to project the source image onto the face.
        audioproj (nn.Module): The AudioProjector model used to project the audio embeddings onto the face.
    """
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        audioproj,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.audioproj = audioproj

    def forward(self,):
        """
        empty function to override abstract function of nn Module
        """

    def get_modules(self):
        """
        Simple method to avoid too-few-public-methods pylint error
        """
        return {
            "reference_unet": self.reference_unet,
            "denoising_unet": self.denoising_unet,
            "pose_guider": self.pose_guider,
            "audioproj": self.audioproj,
        }

def find_best_slice(slice_candidates, last_half):
    last_pos = last_half[-5:] # 5,C
    last_v = last_half[1:] - last_half[:-1] # 39,C
    last_v = np.mean(last_v[-5:], axis=0).reshape(-1,2) # 5,C -> C -> 100,2
    
    min_score = 1000000000
    best_cand = None
    for idx, cand in enumerate(slice_candidates):
        cand_half = cand[:] 
        cand_pos = cand_half[:5] 
        cand_v = cand_half[1:] - cand_half[:-1]
        cand_v = np.mean(cand_v[-5:], axis=0).reshape(-1,2) 
        
        def v_angle_score(array1, array2): # 100,2
            dot_products = np.sum(array1 * array2, axis=1)  
            norms = np.linalg.norm(array1, axis=1) * np.linalg.norm(array2, axis=1)
            cosine_similarity = dot_products / norms
            cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
            angles = np.arccos(cosine_similarity)
            return np.mean(angles)
            
        pos_score = np.sum(np.abs(cand_pos - last_pos))
        v_score = v_angle_score(cand_v*1000, last_v*1000)
        
        final_score = pos_score + v_score

        if final_score < min_score:
            min_score = final_score
            best_cand = cand
    return best_cand


def process_audio_emb(audio_emb):
    """
    Process the audio embedding to concatenate with other tensors.

    Parameters:
        audio_emb (torch.Tensor): The audio embedding tensor to process.

    Returns:
        concatenated_tensors (List[torch.Tensor]): The concatenated tensor list.
    """
    concatenated_tensors = []

    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0]-1), 0)]for j in range(-2, 3)]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))

    audio_emb = torch.stack(concatenated_tensors, dim=0)

    return audio_emb


def blur_mask(mask, resize_dim=(64, 64), kernel_size=(51, 51)):
    """
    Read, resize, blur, normalize, and save an image.

    Parameters:
    file_path (str): Path to the input image file.
    resize_dim (tuple): Dimensions to resize the images to.
    kernel_size (tuple): Size of the kernel to use for Gaussian blur.
    """
    # Check if the image is loaded successfully
    normalized_mask = None
    if mask is not None:
        # Resize the mask image
        resized_mask = cv2.resize(mask, resize_dim)
        # Apply Gaussian blur to the resized mask image
        blurred_mask = cv2.GaussianBlur(resized_mask, kernel_size, 0)
        # Normalize the blurred image
        normalized_mask = cv2.normalize(
            blurred_mask, None, 0, 255, cv2.NORM_MINMAX)
        # Save the normalized mask image
    return normalized_mask

def get_poses_paths(source_dir, parallelism, rank):
    # 将 source_dir 转换为 Path 对象
    source_dir = Path(source_dir)
    
    video_paths = [item for item in sorted(source_dir.iterdir()) if item.is_file() and item.suffix == '.mp4']
    return [video_paths[i] for i in range(len(video_paths)) if i % parallelism == rank]

def get_image_paths(source_dir: Path, parallelism: int, rank: int) -> List[Path]:
    # 将 source_dir 转换为 Path 对象
    source_dir = Path(source_dir)
    
    video_paths = [item for item in sorted(source_dir.iterdir()) if item.is_file() and item.suffix == '.png']
    return [video_paths[i] for i in range(len(video_paths)) if i % parallelism == rank]

def get_wavs_paths(source_dir: Path, parallelism: int, rank: int) -> List[Path]:
    # 将 source_dir 转换为 Path 对象
    source_dir = Path(source_dir)
    
    video_paths = [item for item in sorted(source_dir.iterdir()) if item.is_file() and item.suffix == '.wav']
    return [video_paths[i] for i in range(len(video_paths)) if i % parallelism == rank]

def get_audio_duration(wav_file):
    y, sr = librosa.load(wav_file, sr=None)
    return librosa.get_duration(y=y, sr=sr)

def stringintkey(s):
    return list(map(int, re.findall(r'\d+', s)))



def main(args: argparse.Namespace):
    config = OmegaConf.load(args.config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    audio_ckpt_dir = config.audio_ckpt_dir
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    audio2pose_model = SMGA(args.feature_type, args.motion_diffusion_ckpt)  # 模型初始化
    audio2pose_model.eval()
    wavlm_model, wavlm_cfg = wavlm_init()
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device="cuda")

    infer_config = OmegaConf.load(config.inference_config)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        os.path.join(audio_ckpt_dir, f"net-{args.num_c}.pth"),
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device="cuda")

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    audio_proj = AudioProjModel(
        seq_len=5,
        blocks=12,
        channels=768,
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
    ).to(device="cuda", dtype=weight_dtype)

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)
    generator = torch.manual_seed(args.seed)
    width, height = args.W, args.H

    # Freeze
    vae.requires_grad_(False)  # type: ignore
    reference_unet.requires_grad_(False)  # type: ignore
    denoising_unet.requires_grad_(False)
    audio_proj.requires_grad_(False)
    image_enc.requires_grad_(False)
    pose_guider.requires_grad_(False)

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )

    reference_unet.enable_gradient_checkpointing()  # type: ignore
    denoising_unet.enable_gradient_checkpointing()

    net = Net(
        reference_unet,  # type: ignore
        denoising_unet,
        pose_guider,
        audio_proj,
    )

    m, u = net.load_state_dict(
        torch.load(
            os.path.join(audio_ckpt_dir, "modules", f"net-{args.num_c}.pth"),
            map_location="cpu",
        ),
    )
    assert len(m) == 0 and len(u) == 0, "Fail to load correct checkpoint."
    print("loaded weight from ", os.path.join(audio_ckpt_dir, "motion_module.pth"))

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    # ========= 单样本路径 =========
    source_image_path = args.image_path
    driving_audio = args.audio_path

    basename = Path(source_image_path).stem
    audio_name = Path(driving_audio).stem

    # ========= 临时/输出目录 =========
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--seed_{args.seed}-{args.W}x{args.H}"
    save_dir = Path(f"output/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)

    # ========= 音频切片（>3.3s 时）并提取 A2P 条件 =========
    wav_duration = get_audio_duration(driving_audio)
    if wav_duration > 3.3:
        slice_dir = save_dir / "slices"
        slice_dir.mkdir(parents=True, exist_ok=True)
        if not any(slice_dir.glob("*.wav")):
            print(f"Slicing {driving_audio}")
            slice_audio(driving_audio, 3.2, 3.2, slice_dir)
        file_list = sorted(slice_dir.glob("*.wav"), key=lambda p: stringintkey(str(p)))
    else:
        file_list = [Path(driving_audio)]

    cond_list = []
    for file in tqdm(file_list, desc="Extracting audio2pose features"):
        wavlm_feats, _ = wavlm_extract(wavlm_model, wavlm_cfg, str(file))
        baseline_feats, _ = baseline_extract(str(file))
        cond_list.append(np.concatenate((wavlm_feats, baseline_feats), axis=1))
    cond_list = torch.from_numpy(np.array(cond_list))

    # ========= 从单张图片得到起始姿态 =========
    init_feature = process_reference_image(source_image_path)
    init_feature = np.expand_dims(init_feature, axis=0)
    init_feature = mask_leg(init_feature)

    # ========= 生成整段 TPS（音频->姿态）=========
    tps_result = []
    for index, cond in enumerate(cond_list):
        if index == 0:
            last_frame = init_feature
            slice_result = audio2pose_model.render_sample(
                cond_frame=torch.from_numpy(last_frame).float(), cond=cond, last_half=None, mode='normal'
            )
            slice_result = slice_result.squeeze().cpu().numpy()
        else:
            last_frame = tps_result[-1][59]
            if args.use_motion_selection:
                slice_candidates = []
                for _ in range(5):
                    sc = audio2pose_model.render_sample(
                        cond_frame=torch.from_numpy(last_frame).float(), cond=cond, last_half=None, mode='normal'
                    )
                    sc = sc.squeeze().cpu().numpy()
                    slice_candidates.append(sc)
                slice_result = find_best_slice(slice_candidates, tps_result[-1])
            else:
                slice_result = audio2pose_model.render_sample(
                    cond_frame=torch.from_numpy(last_frame).float(), cond=cond, last_half=None, mode='normal'
                )
                slice_result = slice_result.squeeze().cpu().numpy()
        tps_result.append(slice_result)

    # 拼接 + 平滑
    tps_concat_result = tps_result[0]
    for i in range(1, len(tps_result)):
        tps_concat_result = np.concatenate((tps_concat_result, tps_result[i]), axis=0)  # T, 402

    tps_origin = tps_concat_result
    init_feature_1 = init_feature.astype(np.float32)
    tps_concat_result = tps_concat_result[:-1, :]
    tps_origin = np.concatenate([init_feature_1, tps_concat_result], axis=0)
    tps_smoothed = tps_origin.copy()

    # 平滑（插值）
    smooth_method = 'interpolate'
    if smooth_method == 'interpolate':
        T = tps_origin.shape[0]
        mutation_points = np.arange(60, T, 60)
        for point in mutation_points:
            start_idx = max(0, point - 5)
            end_idx = min(T, point + 5)
            x = list(np.arange(start_idx - 3, start_idx)) + list(np.arange(end_idx, end_idx + 3))
            y = tps_smoothed[x]
            cs = CubicSpline(x, y, axis=0)
            xx = np.arange(start_idx - 2, end_idx + 2)
            interpolated_values = cs(xx)
            tps_smoothed[start_idx - 2:end_idx + 2] = interpolated_values
    torch.cuda.empty_cache()

    # ========= 一次性生成姿态/掩码视频 =========
    full_save_path = os.path.join(args.tem_dir, f"{args.num_epoch}")
    os.makedirs(full_save_path, exist_ok=True)
    out_path_dwpose = os.path.join(full_save_path, "dwpose", f"{audio_name}.mp4")
    out_path_face = os.path.join(full_save_path, "face", f"{audio_name}.mp4")
    out_path_lips = os.path.join(full_save_path, "lips", f"{audio_name}.mp4")
    out_hands_path = os.path.join(full_save_path, "hands", f"{audio_name}.mp4")
    for p in [out_path_dwpose, out_path_face, out_path_lips, out_hands_path]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    pose_vid_generator(tps_smoothed, out_path_dwpose, out_hands_path, out_path_lips, out_path_face, fps=25)

    # ========= 准备管线输入 =========
    driving_audio_path = str(driving_audio)
    driving_pose_path = str(out_path_dwpose)
    face_mask_path = str(out_path_face)
    lips_mask_path = str(out_path_lips)
    ref_image_pil = Image.open(source_image_path).convert("RGB")

    save_path = os.path.join(args.out_dir, f"multi_person_{args.num_c}")
    os.makedirs(save_path, exist_ok=True)
    video_path = os.path.join(save_path, f"{basename}.mp4")
    if os.path.exists(video_path):
        print("该视频已经存在，直接跳过生成。")
        return

    motion_scale = [config.pose_weight, config.face_weight, config.lip_weight]
    img_size = (config.data.source_image.width, config.data.source_image.height)
    image_processor = ImageProcessor(img_size)

    # 3.2 音频特征（驱动合成）
    sample_rate = config.data.driving_audio.sample_rate
    assert sample_rate == 16000, "audio sample rate must be 16000"
    fps_cfg = config.data.export_video.fps
    wav2vec_model_path = config.wav2vec.model_path
    wav2vec_only_last_features = config.wav2vec.features == "last"
    audio_separator_model_file = config.audio_separator.model_path
    with AudioProcessor(
        sample_rate,
        fps_cfg,
        wav2vec_model_path,
        wav2vec_only_last_features,
        os.path.dirname(audio_separator_model_file),
        os.path.basename(audio_separator_model_file),
        os.path.join(save_path, "audio_preprocess")
    ) as audio_processor:
        # 注意：这里的 L 可能会被下调
        audio_emb, audio_length = audio_processor.preprocess(driving_audio_path, args.L)

    audio_emb = process_audio_emb(audio_emb)

    # 读取姿态/掩码帧并对齐长度
    pose_images = read_frames(driving_pose_path)
    face_mask_frames = read_frames(face_mask_path)
    lips_mask_frames = read_frames(lips_mask_path)
    src_fps = get_fps(driving_pose_path)

    avail_len = min(len(pose_images), len(face_mask_frames), len(lips_mask_frames), audio_emb.shape[0])
    if avail_len < args.L:
        print(f"警告：可用帧数 {avail_len} < 期望 L={args.L}，将使用 {avail_len}。")
    L = min(args.L, avail_len)

    # 裁切/组装
    audio_tensor = audio_emb[:L].unsqueeze(0)
    audio_tensor = audio_tensor.to(device=net.audioproj.device, dtype=net.audioproj.dtype)
    audio_tensor = net.audioproj(audio_tensor)

    pose_transform = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])
    pose_tensor_list, pose_list = [], []
    for pose_image_pil in pose_images[:L]:
        pose_tensor_list.append(pose_transform(pose_image_pil))
        pose_list.append(pose_image_pil)

    # 掩码列表
    face_mask_list = face_mask_frames[:L]
    lips_mask_list = lips_mask_frames[:L]

    # 模糊/缩放掩码
    face_masks_list = []
    for face_mask_image in face_mask_list:
        mask_array = np.array(face_mask_image)
        processed_mask = blur_mask(mask_array, resize_dim=(64, 64), kernel_size=(31, 31))
        face_masks_list.append(Image.fromarray(processed_mask.astype(np.uint8)).convert("L"))

    lips_masks_list = []
    for lips_mask_image in lips_mask_list:
        mask_array = np.array(lips_mask_image)
        processed_mask = blur_mask(mask_array, resize_dim=(64, 64), kernel_size=(21, 21))
        lips_masks_list.append(Image.fromarray(processed_mask.astype(np.uint8)).convert("L"))

    pixel_values_face_mask, pixel_values_lips_mask = image_processor.preprocess_mov_mask(
        face_masks_list, lips_masks_list, config.face_expand_ratio, L
    )

    # 构造 full mask
    source_image_full_mask = []
    for mask in pixel_values_face_mask:
        inverted_mask = 1.0 - mask.float()
        source_image_full_mask.append(inverted_mask)
    for i in range(len(pixel_values_lips_mask)):
        source_image_full_mask[i] = 1.0 + pixel_values_lips_mask[i].float()

    ref_image_tensor = pose_transform(ref_image_pil).unsqueeze(1).unsqueeze(0)  # (1,c,1,h,w)
    ref_image_tensor = repeat(ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=L)

    pose_tensor = torch.stack(pose_tensor_list, dim=0).transpose(0, 1).unsqueeze(0)

    # ========= 推理并保存 =========
    pipeline_output = pipe(
        ref_image=ref_image_pil,
        pose_images=pose_list,
        audio_tensor=audio_tensor,
        pixel_values_full_mask=source_image_full_mask,
        pixel_values_face_mask=pixel_values_face_mask,
        pixel_values_lip_mask=pixel_values_lips_mask,
        width=width,
        height=height,
        video_length=L,
        num_inference_steps=config.inference_steps,
        guidance_scale=config.cfg_scale,
        generator=generator,
        motion_scale=motion_scale,
    ).videos

    save_videos_grid(
        pipeline_output,
        video_path,
        n_rows=1,
        fps=src_fps if args.fps is None else args.fps,
    )
    print("Saved:", video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="./configs/prompts/animation.yaml")
    parser.add_argument("--image_path", type=str, required=True, help="单张源图片(.png/.jpg)")
    parser.add_argument("--audio_path", type=str, required=True, help="单段驱动音频(.wav)")
    parser.add_argument("--out_dir", type=str, default="scripts/output_videos", required=False)
    parser.add_argument("--tem_dir", type=str, default="scripts/output_videos/temp")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=80)
    parser.add_argument("--name", default="baseline_pose")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--num_c", type=int, default=32500, help="weight of lip", required=False)
    parser.add_argument("--use_motion_selection", default=False, action='store_true', help="use motion selection")
    parser.add_argument("--num_epoch", type=int, default=3400)
    parser.add_argument("--feature_type", type=str, default="wavlm", help="'baseline' or 'wavlm'")
    parser.add_argument("--motion_diffusion_ckpt", type=str, default="./pretrained_weights/MMGT_pretrained/stage_1/audio2pose_best_model.pt",
                        help="motion diffusion checkpoint")
    args = parser.parse_args()
    main(args)