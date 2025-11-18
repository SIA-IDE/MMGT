# pylint: disable=R0801
"""
This module is used to extract meta information from video files and store them in a JSON file.

The script takes in command line arguments to specify the root path of the video files,
the dataset name, and the name of the meta information file. It then generates a list of
dictionaries containing the meta information for each video file and writes it to a JSON
file with the specified name.

The meta information includes the path to the video file, the mask path, the face mask
path, the face mask union path, the face mask gaussian path, the lip mask path, the lip
mask union path, the lip mask gaussian path, the separate mask border, the separate mask
face, the separate mask lip, the face embedding path, the audio path, the vocals embedding
base last path, the vocals embedding base all path, the vocals embedding base average
path, the vocals embedding large last path, the vocals embedding large all path, and the
vocals embedding large average path.

The script checks if the mask path exists before adding the information to the list.

Usage:
    python scripts/extract_meta_info_stage1.py -r /root/node03-nfs/aaai/hallo/dataset_process/output_lgci/train -n new_LGCI

Example:
    python scripts/extract_meta_info_stage2_move_mask.py --root_path /root/node03-nfs/aaai/hallo/dataset_process/output_lgci/train --dataset_name my_dataset --meta_info_name new_LGCI
"""

import argparse
import json
import os
from pathlib import Path

import torch
from decord import VideoReader, cpu
from tqdm import tqdm
import soundfile as sf

def get_video_paths(root_path: Path, extensions: list) -> list:
    """
    Get a list of video paths from the root path with the specified extensions.

    Args:
        root_path (Path): The root directory containing video files.
        extensions (list): List of file extensions to include.

    Returns:
        list: List of video file paths.
    """
    return [str(path.resolve()) for path in root_path.iterdir() if path.suffix in extensions]


def file_exists(file_path: str) -> bool:
    """
    Check if a file exists.

    Args:
        file_path (str): The path to the file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return os.path.exists(file_path)


def construct_paths(video_path: str, base_dir: str, new_dir: str, new_ext: str) -> str:
    """
    Construct a new path by replacing the base directory and extension in the original path.

    Args:
        video_path (str): The original video path.
        base_dir (str): The base directory to be replaced.
        new_dir (str): The new directory to replace the base directory.
        new_ext (str): The new file extension.

    Returns:
        str: The constructed path.
    """
    return str(video_path).replace(base_dir, new_dir).replace(".mp4", new_ext)


def extract_meta_info(video_path: str) -> dict:
    """
    Extract meta information for a given video file, and if needed,
    trim audio embeddings and corresponding .wav so that they align
    with the number of video frames.

    Args:
        video_path (str): The path to the video file.

    Returns:
        dict or None: A dictionary containing the meta-information for the video,
                      possibly with trimmed audio paths, or None if key files missing.
    """
    # --- 1. 构建各类路径 ---
    kps_path = construct_paths(video_path, "videos", "videos_dwpose", ".mp4")
    sep_mask_face = construct_paths(video_path, "videos", "sep_face_mask", ".mp4")
    sep_mask_lip  = construct_paths(video_path, "videos", "sep_body_mask", ".mp4")
    images_path   = construct_paths(video_path, "videos", "images", "")
    audio_path    = construct_paths(video_path, "videos", "audios", "_audio.wav")
    emb_path      = construct_paths(video_path, "videos", "audio_emb", ".pt")

    # --- 2. 检查所有必须文件是否存在 ---
    assert_flag = True
    for name, p in [
        ("keypoints", kps_path),
        ("face mask", sep_mask_face),
        ("lip mask", sep_mask_lip),
        ("images", images_path),
        ("audio", audio_path),
        ("audio emb", emb_path),
    ]:
        if not file_exists(p):
            print(f"[ERROR] {name} not found: {p}")
            assert_flag = False

    if not assert_flag:
        return None

    # --- 3. 读取视频帧数和原始 audio embedding ---
    video_frames = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(video_frames)
    audio_emb = torch.load(emb_path)
    num_emb = audio_emb.shape[0]
    diff = num_emb - num_frames

    # # --- 4. 如果不匹配，进行裁剪或 pad，并保存到新目录 ---
    if diff != 0:
        print(f"[WARN] frame mismatch: video={num_frames}, emb={num_emb} (diff={diff})")

        # 4.1 裁剪或 pad embedding
        if diff > 0:
            trimmed_emb = audio_emb[diff:]              # 去掉前 diff 帧
        else:
            # audio_emb 太短，前面 pad 零向量
            pad_shape = (-diff, *audio_emb.shape[1:])
            pad = torch.zeros(pad_shape, dtype=audio_emb.dtype)
            trimmed_emb = torch.cat([pad, audio_emb], dim=0)

        # # 4.2 保存新的 embedding
        emb_dir = Path(emb_path).parent
        trimmed_emb_dir = emb_dir / "trimmed_embs"
        trimmed_emb_dir.mkdir(exist_ok=True)
        new_emb_path = trimmed_emb_dir / Path(emb_path).name
        if file_exists(new_emb_path):
            print(f"[WARN] trimmed emb already exists: {new_emb_path}")
            return None
        # torch.save(trimmed_emb, new_emb_path)
        print(f"[INFO] saved trimmed emb -> {new_emb_path}")
        emb_path = str(new_emb_path)

        # 4.3 裁剪对应的 .wav
        wav_dir = Path(audio_path).parent
        trimmed_wav_dir = wav_dir / "trimmed_wavs"
        trimmed_wav_dir.mkdir(exist_ok=True)

        audio, sr = sf.read(audio_path)                  # (n_samples, ...)
        samples_per_emb = len(audio) // num_emb
        trim_samples = max(diff, 0) * samples_per_emb
        new_audio = audio[trim_samples:]                 # 从头去掉多余采样
        new_wav_path = trimmed_wav_dir / Path(audio_path).name
        if file_exists(new_wav_path):
            print(f"[WARN] trimmed wav already exists: {new_wav_path}")
            return None
        # sf.write(str(new_wav_path), new_audio, sr)
        print(f"[INFO] saved trimmed wav -> {new_wav_path}")
        audio_path = str(new_wav_path)

    # 释放资源
    del video_frames, audio_emb

    # --- 5. 返回最终信息 ---
    return {
        "video_path": str(video_path),
        "kps_path": kps_path,
        "face_mask_path": sep_mask_face,
        "lip_mask_path": sep_mask_lip,
        "images_path": images_path,
        "audio_path": audio_path,
        "vocals_emb_base_all": emb_path,
    }


def main():
    """
    Main function to extract meta info for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_path", type=str,
                        required=True, help="Root path of the video files")
    parser.add_argument("-n", "--dataset_name", type=str,
                        required=True, help="Name of the dataset")
    parser.add_argument("--meta_info_name", type=str,
                        help="Name of the meta information file")

    args = parser.parse_args()

    if args.meta_info_name is None:
        args.meta_info_name = args.dataset_name

    video_dir = Path(args.root_path) / "videos"
    video_paths = get_video_paths(video_dir, [".mp4"])

    meta_infos = []

    for video_path in tqdm(video_paths, desc="Extracting meta info"):
        try:
            # 核心修改点：添加try-except结构
            meta_info = extract_meta_info(video_path)
            if meta_info:
                meta_infos.append(meta_info)
                
        except Exception as e:
            # 详细错误日志记录
            error_msg = f"处理视频失败: {video_path}\n错误类型: {type(e).__name__}\n错误详情: {str(e)}"
            print(f"\n{'-'*40}")
            print(error_msg)
            print(f"{'-'*40}\n")
            
    print(f"Final data count: {len(meta_infos)}")

    output_file = Path(f"./data/{args.meta_info_name}_stage2.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(meta_infos, f, indent=4)


if __name__ == "__main__":
    main()
