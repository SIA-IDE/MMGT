import glob
import os
import pickle
import cv2

import librosa as lr
import numpy as np
import soundfile as sf
from tqdm import tqdm


def slice_audio(audio_file, stride, length, out_dir):
    # stride, length in seconds
    audio, sr = lr.load(audio_file, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_idx = 0
    idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    while start_idx <= len(audio) - window:
        if start_idx == 0:
            start_idx += stride_step
        else:
            audio_slice = audio[start_idx : start_idx + window]
            sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
            start_idx += stride_step
            idx += 1
    return idx


def slice_keypoint(keypoint_file, stride, length, num_slices, out_dir):
    keypoint = np.load(keypoint_file)
    file_name = os.path.splitext(os.path.basename(keypoint_file))[0]
    start_idx = 0
    window = int(length * 24)
    stride_step = int(stride * 24)
    slice_count = 0
    # slice until done or until matching audio slices
    while start_idx <= len(keypoint) - window and slice_count < num_slices:
        if start_idx == 0:
            start_idx += stride_step
        else:
            # save the first frame as condition
            keypoint_slice = keypoint[start_idx - 1 : start_idx + window]
            np.save(f"{out_dir}/{file_name}_slice{slice_count}.npy", keypoint_slice)
            start_idx += stride_step
            slice_count += 1
    return slice_count

def slice_wavlm(wavlm_file, stride, length, num_slices, out_dir):
    # stride, length in seconds
    wavlm = np.load(wavlm_file)
    file_name = os.path.splitext(os.path.basename(wavlm_file))[0]
    start_idx = 0
    slice_count = 0
    window = int(length * 24)
    stride_step = int(stride * 24)
    while start_idx <= len(wavlm) - window and slice_count < num_slices:
        if start_idx == 0:
            start_idx += stride_step
        else:
            wavlm_slice = wavlm[start_idx : start_idx + window]
            np.save(f"{out_dir}/{file_name}_slice{slice_count}.npy", wavlm_slice)
            start_idx += stride_step
            slice_count += 1
    return slice_count
        
import os
import glob
from tqdm import tqdm

def slice_data_1(keypoint_dir, wav_dir, wavlm_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    keypoints = sorted(glob.glob(f"{keypoint_dir}/*.npy"))
    wavlm = sorted(glob.glob(f"{wavlm_dir}/*.npy"))
    wav_out = wav_dir + "_sliced"
    keypoint_out = keypoint_dir + "_sliced"
    wavlm_out = wavlm_dir + "_sliced"
    
    # Create directories for sliced data if they don't exist
    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(keypoint_out, exist_ok=True)
    os.makedirs(wavlm_out, exist_ok=True)
    
    # Log the number of files found
    print(f"Found {len(wavs)} audio files, {len(keypoints)} keypoint files, and {len(wavlm)} wavlm files.")
    
    # Ensure the input directories have matching file counts
    assert len(wavs) == len(keypoints) == len(wavlm), "Mismatched number of files in directories."

    # Loop through all videos and process them
    for wav, keypoint, wavlm in tqdm(zip(wavs, keypoints, wavlm), desc="Processing videos"):
        # Extract file names without extensions
        m_name = os.path.splitext(os.path.basename(keypoint))[0]
        w_name = os.path.splitext(os.path.basename(wav))[0]
        wavlm_name = os.path.splitext(os.path.basename(wavlm))[0]
        
        # Ensure the file names match
        assert m_name == w_name == wavlm_name, f"File name mismatch: {keypoint}, {wav}, {wavlm}"

        # Check if the output files already exist
        output_audio_path = os.path.join(wav_out, f"{w_name}_sliced.wav")
        output_keypoint_path = os.path.join(keypoint_out, f"{m_name}_sliced.npy")
        output_wavlm_path = os.path.join(wavlm_out, f"{wavlm_name}_sliced.npy")
        
        if os.path.exists(output_audio_path) and os.path.exists(output_keypoint_path) and os.path.exists(output_wavlm_path):
            print(f"Skipping {w_name} - already processed.")
            continue  # Skip this video if all output files exist
        
        # Log the processing start
        print(f"Processing {w_name}...")

        # Perform slicing and save the slices
        try:
            audio_slices = slice_audio(wav, stride, length, wav_out)
            keypoint_slices = slice_keypoint(keypoint, stride, length, audio_slices, keypoint_out)
            wavlm_slices = slice_wavlm(wavlm, stride, length, audio_slices, wavlm_out)

            # Ensure the slices are consistent
            assert audio_slices == keypoint_slices == wavlm_slices, f"Slice mismatch for {w_name}: {audio_slices}, {keypoint_slices}, {wavlm_slices}"
            print(f"Successfully processed {w_name}.")
        
        except Exception as e:
            print(f"Error processing {w_name}: {e}")


def slice_data(keypoint_dir, wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    keypoints = sorted(glob.glob(f"{keypoint_dir}/*.npy"))
    # wavlm = sorted(glob.glob(f"{wavlm_dir}/*.npy"))
    wav_out = wav_dir + "_sliced"
    keypoint_out = keypoint_dir + "_sliced"
    # wavlm_out = wavlm_dir + "_sliced"
    
    # Create directories for sliced data if they don't exist
    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(keypoint_out, exist_ok=True)
    # os.makedirs(wavlm_out, exist_ok=True)
    
    # Log the number of files found
    print(f"Found {len(wavs)} audio files, {len(keypoints)} keypoint files, andwavlm files.")
    
    # Ensure the input directories have matching file counts
    assert len(wavs) == len(keypoints), "Mismatched number of files in directories."

    # Loop through all videos and process them
    for wav, keypoint in tqdm(zip(wavs, keypoints), desc="Processing videos"):
        # Extract file names without extensions
        m_name = os.path.splitext(os.path.basename(keypoint))[0]
        w_name = os.path.splitext(os.path.basename(wav))[0]
        # wavlm_name = os.path.splitext(os.path.basename(wavlm))[0]
        
        # Ensure the file names match
        assert m_name == w_name, f"File name mismatch: {keypoint}, {wav}"

        # Check if the output files already exist
        output_audio_path = os.path.join(wav_out, f"{w_name}_sliced.wav")
        output_keypoint_path = os.path.join(keypoint_out, f"{m_name}_sliced.npy")
        # output_wavlm_path = os.path.join(wavlm_out, f"{wavlm_name}_sliced.npy")
        
        if os.path.exists(output_audio_path) and os.path.exists(output_keypoint_path) :
            print(f"Skipping {w_name} - already processed.")
            continue  # Skip this video if all output files exist
        
        # Log the processing start
        print(f"Processing {w_name}...")

        # Perform slicing and save the slices
        try:
            audio_slices = slice_audio(wav, stride, length, wav_out)
            keypoint_slices = slice_keypoint(keypoint, stride, length, audio_slices, keypoint_out)
            # wavlm_slices = slice_wavlm(wavlm, stride, length, audio_slices, wavlm_out)

            # Ensure the slices are consistent
            assert audio_slices == keypoint_slices, f"Slice mismatch for {w_name}: {audio_slices}, {keypoint_slices}"
            print(f"Successfully processed {w_name}.")
        
        except Exception as e:
            print(f"Error processing {w_name}: {e}")


# def slice_data_video(wav_dir, video_dir, video_dwpose_dir, keypoint_dir, wavlm_dir, stride=0.5, length=5):
#     wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
#     keypoints = sorted(glob.glob(f"{keypoint_dir}/*.npy"))
#     wavlm = sorted(glob.glob(f"{wavlm_dir}/*.npy"))
#     videos = sorted(glob.glob(f"{video_dir}/*.mp4") + glob.glob(f"{video_dir}/*.avi"))
#     video_dwpose = sorted(glob.glob(f"{video_dwpose_dir}/*.mp4") + glob.glob(f"{video_dwpose_dir}/*.avi"))
    
#     wav_out = wav_dir + "_sliced"
#     keypoint_out = keypoint_dir + "_sliced"
#     wavlm_out = wavlm_dir + "_sliced"
#     video_out = video_dir + "_sliced"
#     video_dwpose_out = video_dwpose_dir + "_sliced"
    
#     os.makedirs(wav_out, exist_ok=True)
#     os.makedirs(keypoint_out, exist_ok=True)
#     os.makedirs(wavlm_out, exist_ok=True)
#     os.makedirs(video_out, exist_ok=True)
#     os.makedirs(video_dwpose_out, exist_ok=True)

#     print(len(wavs))
#     print(len(keypoints))
#     print(len(wavlm))
#     print(len(videos))
#     print(len(video_dwpose))  
#     assert len(wavs) == len(keypoints) == len(wavlm) == len(videos)
     
#     for keypoints, wav,  wavlm, video, video_dwpose in tqdm(zip(keypoints, wavs, wavlm, videos, video_dwpose)):
#         # 确保文件名匹配
#         m_name = os.path.splitext(os.path.basename(keypoints))[0]
#         w_name = os.path.splitext(os.path.basename(wav))[0]
#         wavlm_name = os.path.splitext(os.path.basename(wavlm))[0]
#         video_name = os.path.splitext(os.path.basename(video))[0]
#         video_dwpose_name = os.path.splitext(os.path.basename(video_dwpose))[0]
#         assert m_name == w_name == wavlm_name == video_name == video_dwpose_name, str((keypoints, wav, wavlm_name, video_name, video_dwpose_name))
        
#         # 对音频进行切片
#         audio_slices = slice_audio(wav, stride, length, wav_out)
#         keypoint_slices = slice_keypoint(keypoints, stride, length, audio_slices, keypoint_out)
#         wavlm_slices = slice_wavlm(wavlm, stride, length, audio_slices, wavlm_out)
        
#         # 对视频进行切片
#         video_slices = slice_video(video, stride, length, audio_slices, video_out)

#         video_dwpose_slices = slice_video(video_dwpose, stride, length, audio_slices, video_dwpose_out)
        
#         # 确保所有切片的数量保持一致
#         assert audio_slices == video_slices == video_dwpose_slices == keypoint_slices == wavlm_slices, str(
#             (wav, video, video_dwpose, audio_slices, video_slices, video_dwpose_slices)
#         )



def slice_data_video(wav_dir, video_dir, video_dwpose_dir, keypoint_dir, wavlm_dir, stride=0.5, length=5):
    # Get the list of files
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    keypoints = sorted(glob.glob(f"{keypoint_dir}/*.npy"))
    # wavlm = sorted(glob.glob(f"{wavlm_dir}/*.npy"))
    videos = sorted(glob.glob(f"{video_dir}/*.mp4") + glob.glob(f"{video_dir}/*.avi"))
    video_dwpose = sorted(glob.glob(f"{video_dwpose_dir}/*.mp4") + glob.glob(f"{video_dwpose_dir}/*.avi"))
    
    # Output directories
    wav_out = wav_dir + "_sliced"
    keypoint_out = keypoint_dir + "_sliced"
    # wavlm_out = wavlm_dir + "_sliced"
    video_out = video_dir + "_sliced"
    video_dwpose_out = video_dwpose_dir + "_sliced"
    
    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(keypoint_out, exist_ok=True)
    # os.makedirs(wavlm_out, exist_ok=True)
    os.makedirs(video_out, exist_ok=True)
    os.makedirs(video_dwpose_out, exist_ok=True)

    # Log the number of files
    print(f"Found {len(wavs)} wav files, {len(keypoints)} keypoints, {len(videos)} videos, and {len(video_dwpose)} video_dwpose files.")
    
    # Ensure the number of input files match
    assert len(wavs) == len(keypoints) == len(videos) == len(video_dwpose), "Input files are not aligned."

    # Process each file
    for keypoints, wav, video, video_dwpose in tqdm(zip(keypoints, wavs, videos, video_dwpose), desc="Processing videos"):
        # Get the base name of each file
        m_name = os.path.splitext(os.path.basename(keypoints))[0]
        w_name = os.path.splitext(os.path.basename(wav))[0]
        # wavlm_name = os.path.splitext(os.path.basename(wavlm))[0]
        video_name = os.path.splitext(os.path.basename(video))[0]
        video_dwpose_name = os.path.splitext(os.path.basename(video_dwpose))[0]
        
        # Ensure file names match
        assert m_name == w_name  == video_name == video_dwpose_name, f"File names don't match: {keypoints}, {wav}, {video}, {video_dwpose}"
        
        # Output file paths
        output_audio_path = os.path.join(wav_out, f"{w_name}_sliced.wav")
        output_keypoint_path = os.path.join(keypoint_out, f"{m_name}_sliced.npy")
        # output_wavlm_path = os.path.join(wavlm_out, f"{wavlm_name}_sliced.npy")
        output_video_path = os.path.join(video_out, f"{video_name}_sliced.mp4")
        output_video_dwpose_path = os.path.join(video_dwpose_out, f"{video_dwpose_name}_sliced.mp4")

        output_audio_path_c = os.path.join(wav_out, f"{w_name}_slice0.wav")
        output_keypoint_path_c = os.path.join(keypoint_out, f"{m_name}_slice0.npy")
        # output_wavlm_path = os.path.join(wavlm_out, f"{wavlm_name}_sliced.npy")
        output_video_path_c = os.path.join(video_out, f"{video_name}_slice0.mp4")
        output_video_dwpose_path_c = os.path.join(video_dwpose_out, f"{video_dwpose_name}_slice0.mp4")

        # Check if all output files exist and skip if they do
        if os.path.exists(output_audio_path_c) and os.path.exists(output_keypoint_path_c) and os.path.exists(output_video_path_c) and os.path.exists(output_video_dwpose_path_c):
            print(f"Skipping {w_name} - already processed.")
            continue  # Skip processing this file

        # Proceed with slicing
        try:
            print(f"Processing {w_name}...")
            
            # Perform slicing
            audio_slices = slice_audio(wav, stride, length, wav_out)
            keypoint_slices = slice_keypoint(keypoints, stride, length, audio_slices, keypoint_out)
            # wavlm_slices = slice_wavlm(wavlm, stride, length, audio_slices, wavlm_out)
            video_slices = slice_video(video, stride, length, audio_slices, video_out)
            video_dwpose_slices = slice_video(video_dwpose, stride, length, audio_slices, video_dwpose_out)

            # Ensure all slices are consistent
            assert audio_slices == keypoint_slices == video_slices == video_dwpose_slices, f"Slice mismatch: {w_name}"

            print(f"Successfully processed {w_name}.")
        
        except Exception as e:
            print(f"Error processing {w_name}: {e}")




def slice_audio_folder(wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    wav_out = wav_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    for wav in tqdm(wavs):
        audio_slices = slice_audio(wav, stride, length, wav_out)
        
def slice_wavlm_folder(wavlm_dir, stride, audio_slices, length):
    wavlms = sorted(glob.glob(f"{wavlm_dir}/*.npy"))
    wavlm_out = wavlm_dir[:-4]
    os.makedirs(wavlm_out, exist_ok=True)
    for wavlm in tqdm(wavlms):
        slice_wavlm(wavlm, stride, length, audio_slices, wavlm_out)

def slice_video_folder(video_dir, audio_slices, stride=0.5, length=5):
    # 获取目录中的所有视频文件（假设视频格式为 .mp4 或 .avi）
    videos = sorted(glob.glob(f"{video_dir}/*.mp4") + glob.glob(f"{video_dir}/*.avi"))
    
    # 定义输出目录
    video_out = video_dir + "_sliced"
    os.makedirs(video_out, exist_ok=True)
    
    # 遍历每个视频文件，并进行切片处理
    for video in tqdm(videos):
        # 对每个视频调用切片函数
        slice_video(video, stride, length, audio_slices, video_out)



def slice_video(video_file, stride, length, num_slices, out_dir):
    """
    对视频进行切割，并确保切片数量与音频一致
    :param video_file: 输入的视频文件路径
    :param stride: 切片步长，单位是秒
    :param length: 切片长度，单位是秒
    :param out_dir: 输出文件夹路径，用于保存切片
    :param num_slices: 预期的切片数量（由音频切片数量决定）
    """
    # 获取视频文件名
    file_name = os.path.splitext(os.path.basename(video_file))[0]

    # 使用 OpenCV 打开视频
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"无法打开视频文件 {video_file}")
        return 0

    # 获取视频的帧率 (fps) 和总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 将时间转化为帧数
    window_frames = int(length * fps)  # 每个片段的帧数
    stride_frames = int(stride * fps)  # 步长对应的帧数

    start_idx = 0
    idx = 0

    # 循环切片，确保切片数量不超过 num_slices
    while start_idx <= total_frames - window_frames and idx < num_slices:
        # 创建视频写入对象
        out_video_path = f"{out_dir}/{file_name}_slice{idx}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
        out = None

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)  # 设置视频读取开始的帧位置
        frames_to_write = []

        for _ in range(window_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # 如果没有初始化视频写入器，则创建它
            if out is None:
                frame_height, frame_width = frame.shape[:2]
                out = cv2.VideoWriter(out_video_path, fourcc, fps, (frame_width, frame_height))

            out.write(frame)

        # 如果视频写入器已创建，则释放它
        if out is not None:
            out.release()

        start_idx += stride_frames  # 更新下一次开始的帧索引
        idx += 1

    cap.release()
    return idx