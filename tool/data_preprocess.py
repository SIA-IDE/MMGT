# pylint: disable=W1203,W0718
"""
This module is used to process videos to prepare data for training. It utilizes various libraries and models
to perform tasks such as video frame extraction, audio extraction, face mask generation, and face embedding extraction.
The script takes in command-line arguments to specify the input and output directories, GPU status, level of parallelism,
and rank for distributed processing.

Usage:
    python -m scripts.data_preprocess --input_dir /path/to/video_dir --dataset_name dataset_name --gpu_status --parallelism 4 --rank 0

Example:
    python -m scripts.data_preprocess -i data/videos -o data/output -g -p 4 -r 0
"""
import argparse
import logging
import os
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

from src.dataset.audio_processor import AudioProcessor
from src.dataset.image_processor import ImageProcessorForDataProcessing
from src.utils.util import convert_video_to_images, extract_audio_from_videos
from src.models.pose_guider import PoseGuider

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def setup_directories(video_path: Path) -> dict:
    """
    Setup directories for storing processed files.

    Args:
        video_path (Path): Path to the video file.

    Returns:
        dict: A dictionary containing paths for various directories.
    """
    base_dir = video_path.parent.parent
    dirs = {
        "body_mask": base_dir / "body_mask",
        "sep_pose_mask": base_dir / "sep_pose_mask",
        "sep_face_mask": base_dir / "sep_face_mask",
        "sep_body_mask": base_dir / "sep_body_mask",
        "body_emb": base_dir / "body_emb",
        "audio_emb": base_dir / "audio_emb",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def process_single_video(video_path: Path,
                         output_dir: Path,
                         image_processor: ImageProcessorForDataProcessing,
                         audio_processor: AudioProcessor,
                         step: int) -> None:
    """
    Process a single video file, skipping if the target output already exists.

    Args:
        video_path (Path): Path to the video file.
        output_dir (Path): Directory to save the output.
        image_processor (ImageProcessorForDataProcessing): Image processor object.
        audio_processor (AudioProcessor): Audio processor object.
        step (int): Processing step to control whether images or audio are processed.
    """
    assert video_path.exists(), f"Video path {video_path} does not exist"
    
    # Check if the video has already been processed
    images_dir = output_dir / "images" / video_path.stem
    audio_emb_path = output_dir / "audio_emb" / f"{video_path.stem}.pt"

    
    if step == 1 and images_dir.exists() and any(images_dir.iterdir()):
        logging.info(f"Skipping video {video_path} - images already processed.")
        return
    if step == 2 and audio_emb_path.exists():
        logging.info(f"Skipping video {video_path} - audio embedding already processed.")
        return

    dirs = setup_directories(video_path)
    logging.info(f"Processing video: {video_path}")

    try:
        if step == 1:
            images_output_dir = output_dir / 'images' / video_path.stem
            images_output_dir.mkdir(parents=True, exist_ok=True)
            images_output_dir = convert_video_to_images(video_path, images_output_dir)
            logging.info(f"Images saved to: {images_output_dir}")
        else:
            images_dir = output_dir / "images" / video_path.stem
            audio_path = output_dir / "audios" / f"{video_path.stem}.wav"
            if not os.path.exists(audio_path):
                logging.warning(f"[skip] audio not found: {audio_path}")
                return  # 直接返回，让上层继续处理下一个
            audio_emb_1, _ = audio_processor.preprocess(audio_path)
            torch.save(audio_emb_1, audio_emb_path)
    except Exception as e:
        logging.error(f"Failed to process video {video_path}: {e}")


def process_all_videos(input_video_list: List[Path], output_dir: Path, step: int) -> None:
    """
    Process all videos in the input list.

    Args:
        input_video_list (List[Path]): List of video paths to process.
        output_dir (Path): Directory to save the output.
        gpu_status (bool): Whether to use GPU for processing.
    """
    face_analysis_model_path = "pretrained_models/face_analysis"
    landmark_model_path = "pretrained_models/face_analysis/models/face_landmarker_v2_with_blendshapes.task"
    audio_separator_model_file = "pretrained_models/audio_separator/Kim_Vocal_2.onnx"
    wav2vec_model_path = 'pretrained_models/wav2vec/wav2vec2-base-960h'

    audio_processor = AudioProcessor(
        16000,
        25,
        wav2vec_model_path,
        False,
        os.path.dirname(audio_separator_model_file),
        os.path.basename(audio_separator_model_file),
        os.path.join(output_dir, "vocals"),
    ) if step==2 else None

    image_processor = ImageProcessorForDataProcessing(
        face_analysis_model_path, landmark_model_path, step)

    for video_path in tqdm(input_video_list, desc="Processing videos"):
        process_single_video(video_path, output_dir,
                             image_processor, audio_processor, step)


def get_video_paths(source_dir: Path, parallelism: int, rank: int) -> List[Path]:
    """
    Get paths of videos to process, partitioned for parallel processing.

    Args:
        source_dir (Path): Source directory containing videos.
        parallelism (int): Level of parallelism.
        rank (int): Rank for distributed processing.

    Returns:
        List[Path]: List of video paths to process.
    """
    video_paths = [item for item in sorted(
        source_dir.iterdir()) if item.is_file() and item.suffix == '.mp4']
    return [video_paths[i] for i in range(len(video_paths)) if i % parallelism == rank]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process videos to prepare data for training. Run this script twice with different GPU status parameters."
    )
    parser.add_argument("-i", "--input_dir", type=Path,
                        required=True, help="Directory containing videos")
    parser.add_argument("-o", "--output_dir", type=Path,
                        help="Directory to save results, default is parent dir of input dir")
    parser.add_argument("-s", "--step", type=int, default=1,
                        help="Specify data processing step 1 or 2, you should run 1 and 2 sequently")
    parser.add_argument("-p", "--parallelism", default=1,
                        type=int, help="Level of parallelism")
    parser.add_argument("-r", "--rank", default=0, type=int,
                        help="Rank for distributed processing")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir.parent

    video_path_list = get_video_paths(
        args.input_dir, args.parallelism, args.rank)

    if not video_path_list:
        logging.warning("No videos to process.")
    else:
        process_all_videos(video_path_list, args.output_dir, args.step)



# nohup python -m scripts.data_preprocess --input_dir /root/node03-nfs/cvpr_2026/Cogestrue-Videos/finetune/Disney-VideoGeneration-Dataset/test/videos --step 2 --parallelism 8 --rank 0  > scripts/train_1.log 2>&1 &
# nohup python -m scripts.data_preprocess --input_dir /root/node03-nfs/cvpr_2026/Cogestrue-Videos/finetune/Disney-VideoGeneration-Dataset/LGCV/train/audios --step 2 --parallelism 8 --rank 1   > scripts/train_2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -m scripts.data_preprocess --input_dir /root/node03-nfs/cvpr_2026/Cogestrue-Videos/finetune/Disney-VideoGeneration-Dataset/LGCV/train/audios --step 2 --parallelism 8 --rank 2   > scripts/train_3.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m scripts.data_preprocess --input_dir data/sliced_data/sliced_data/test_2048/videos --step 2 --parallelism 1 --rank 0  