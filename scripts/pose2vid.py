import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List
from tqdm import tqdm
import av
import numpy as np
import torch
import torchvision
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
from torch import nn
from configs.prompts.test_cases import TestCasesDict
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


def main(args: argparse.Namespace):
    config = OmegaConf.load(args.config)
    weight_dtype = torch.float16 if config.weight_dtype == "fp16" else torch.float32

    # --- modules ---
    audio_ckpt_dir = config.audio_ckpt_dir
    vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path, subfolder="unet"
    ).to(dtype=weight_dtype, device="cuda")

    infer_config = OmegaConf.load(config.inference_config)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        os.path.join(audio_ckpt_dir, f"net-{args.num_c}.pth"),
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device="cuda")

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(dtype=weight_dtype, device="cuda")
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    # 仅用于权重加载与维度约束（推理中我们传零向量音频条件）
    audio_proj = AudioProjModel(
        seq_len=5, blocks=12, channels=768, intermediate_dim=512,
        output_dim=768, context_tokens=32
    ).to(device="cuda", dtype=weight_dtype)
    CTOK, CDIM = 32, 768  # 与上面一致

    scheduler = DDIMScheduler(**OmegaConf.to_container(infer_config.noise_scheduler_kwargs))
    generator = torch.manual_seed(args.seed)
    width, height = args.W, args.H

    # --- freeze ---
    for m in [vae, reference_unet, denoising_unet, pose_guider, image_enc, audio_proj]:
        m.requires_grad_(False)

    # --- load weights ---
    denoising_unet.load_state_dict(torch.load(config.denoising_unet_path, map_location="cpu"), strict=False)
    reference_unet.load_state_dict(torch.load(config.reference_unet_path, map_location="cpu"))
    pose_guider.load_state_dict(torch.load(config.pose_guider_path, map_location="cpu"))

    reference_unet.enable_gradient_checkpointing()  # type: ignore
    denoising_unet.enable_gradient_checkpointing()

    net = Net(reference_unet, denoising_unet, pose_guider, audio_proj)
    m, u = net.load_state_dict(
        torch.load(os.path.join(audio_ckpt_dir, "modules", f"net-{args.num_c}.pth"), map_location="cpu"),
    )
    assert len(m) == 0 and len(u) == 0, "Fail to load correct checkpoint."
    print("loaded weight from ", os.path.join(audio_ckpt_dir, "motion_module.pth"))

    pipe = Pose2VideoPipeline(
        vae=vae, image_encoder=image_enc,
        reference_unet=reference_unet, denoising_unet=denoising_unet,
        pose_guider=pose_guider, scheduler=scheduler,
    ).to("cuda", dtype=weight_dtype)

    # -------- 单样本路径 --------
    ref_image_path = args.image_path
    pose_video_path = args.pose_path
    face_mask_path = args.face_mask_path
    lips_mask_path = args.lips_mask_path
    hands_mask_path = args.hands_mask_path  # 可选

    basename = Path(ref_image_path).stem
    save_path = os.path.join(args.out_dir, f"multi_person_{args.num_c}")
    os.makedirs(save_path, exist_ok=True)
    video_path = os.path.join(save_path, f"{basename}.mp4")
    if os.path.exists(video_path):
        print("该视频已存在，跳过生成。")
        return

    # ---- 读取参考图、姿态与掩码 ----
    ref_image_pil = Image.open(ref_image_path).convert("RGB")

    pose_images = read_frames(pose_video_path)
    face_mask_frames = read_frames(face_mask_path)
    lips_mask_frames = read_frames(lips_mask_path)
    hands_mask_frames = read_frames(hands_mask_path) if (hands_mask_path and os.path.exists(hands_mask_path)) else None

    src_fps = get_fps(pose_video_path)
    avail_len = min(len(pose_images), len(face_mask_frames), len(lips_mask_frames),
                    len(hands_mask_frames) if hands_mask_frames else 10**9, args.L)
    if avail_len < 1:
        raise RuntimeError("可用帧数为 0，请检查输入视频/掩码。")
    if avail_len < args.L:
        print(f"提示：可用帧数 {avail_len} 小于 L={args.L}，将使用 {avail_len} 帧。")
    L = avail_len

    # ---- 预处理姿态帧 ----
    pose_transform = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])
    pose_tensor_list, pose_list = [], []
    for pose_image_pil in pose_images[:L]:
        pose_tensor_list.append(pose_transform(pose_image_pil))
        pose_list.append(pose_image_pil)

    # ---- 掩码预处理（模糊->灰度->缩放到 64×64）----
    def _prep_mask_list(frames, ksize):
        out = []
        for img in frames[:L]:
            arr = np.array(img)
            proc = blur_mask(arr, resize_dim=(64, 64), kernel_size=ksize)
            out.append(Image.fromarray(proc.astype(np.uint8)).convert("L"))
        return out

    face_masks_list = _prep_mask_list(face_mask_frames, (31, 31))
    lips_masks_list = _prep_mask_list(lips_mask_frames, (21, 21))
    if hands_mask_frames:
        hands_masks_list = _prep_mask_list(hands_mask_frames, (21, 21))
    else:
        hands_masks_list = [Image.new("L", (64, 64), 0) for _ in range(L)]  # 空手部掩码

    # 统一到管线期望张量
    img_size = (config.data.source_image.width, config.data.source_image.height)
    image_processor = ImageProcessor(img_size)
    pixel_values_face_mask, pixel_values_lips_mask = image_processor.preprocess_mov_mask(
        face_masks_list, lips_masks_list, config.face_expand_ratio, L
    )

    # 手部：用基本 ToTensor 归一化到 [0,1]
    to_tensor_64 = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    pixel_values_hands_mask = [to_tensor_64(m.convert("L")) for m in hands_masks_list]

    # 构造 full mask：1 - face + lips + hands（并裁剪到 [0,1]）
    source_image_full_mask = []
    for i in range(L):
        full = (1.0 - pixel_values_face_mask[i].float()) \
               + pixel_values_lips_mask[i].float() \
               + pixel_values_hands_mask[i].float()
        source_image_full_mask.append(torch.clamp(full, 0.0, 1.0))

    # 参考图重复到 L 帧（仅用于可视化拼接时）
    ref_image_tensor = pose_transform(ref_image_pil).unsqueeze(1).unsqueeze(0)
    ref_image_tensor = repeat(ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=L)
    pose_tensor = torch.stack(pose_tensor_list, dim=0).transpose(0, 1).unsqueeze(0)

    # ---- 零向量音频条件（形状与 AudioProj 输出一致）----
    audio_tensor = torch.zeros((1, L, CTOK, CDIM), dtype=weight_dtype, device="cuda")

    motion_scale = [config.pose_weight, config.face_weight, config.lip_weight]

    # ---- 推理 ----
    pipeline_output = pipe(
        ref_image=ref_image_pil,
        pose_images=pose_list,
        audio_tensor=audio_tensor,
        pixel_values_full_mask=source_image_full_mask,
        pixel_values_face_mask=pixel_values_face_mask,
        pixel_values_lip_mask=pixel_values_lips_mask,
        width=width, height=height, video_length=L,
        num_inference_steps=config.inference_steps,
        guidance_scale=config.cfg_scale,
        generator=generator,
        motion_scale=motion_scale,
    ).videos

    save_videos_grid(
        pipeline_output, video_path, n_rows=1,
        fps=src_fps if args.fps is None else args.fps,
    )
    print("Saved:", video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="./configs/prompts/animation.yaml")
    parser.add_argument("--image_path", type=str, required=True, help="单张源图片路径(.png/.jpg)")
    parser.add_argument("--pose_path", type=str, required=True, help="姿态视频(.mp4)")
    parser.add_argument("--face_mask_path", type=str, required=True, help="脸部掩码视频(.mp4)")
    parser.add_argument("--lips_mask_path", type=str, required=True, help="嘴唇掩码视频(.mp4)")
    parser.add_argument("--hands_mask_path", type=str, default="", help="手部掩码视频(.mp4，可选)")
    parser.add_argument("--out_dir", type=str, default="./outputs", required=False)

    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--num_c", type=int, default=32500)
    args = parser.parse_args()
    main(args)