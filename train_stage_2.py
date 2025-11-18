import argparse
import copy
import logging
import math
import os
import cv2
import os.path as osp
import random
import time
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from src.dataset.audio_processor import AudioProcessor
from src.dataset.image_processor import ImageProcessor
import numpy as np
import matplotlib.pyplot as plt

import diffusers
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from typing import List, Tuple
from src.models.audio_proj import AudioProjModel
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import CLIPVisionModelWithProjection

from src.dataset.dance_video import HumanDanceVideoDataset
from src.dataset.talk_video import TalkingVideoDataset_move_mask
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid import Pose2VideoPipeline
from src.utils.util import (
    init_output_dir,
    delete_additional_ckpt,
    import_filename,
    read_frames,
    save_videos_grid,
    seed_everything,
    tensor_to_video,
)

warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def process_audio_emb(audio_emb: torch.Tensor) -> torch.Tensor:

    concatenated_tensors = []

    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0] - 1), 0)]for j in range(-2, 3)]
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


def get_attention_mask(mask: torch.Tensor, weight_dtype: torch.dtype) :
    if isinstance(mask, List):
        _mask = []
        for m in mask:
            _mask.append(
                rearrange(m, "b f 1 h w -> (b f) (h w)").to(weight_dtype))
        return _mask
    mask = rearrange(mask, "b f 1 h w -> (b f) (h w)").to(weight_dtype)
    return mask

class Net(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
        audioproj,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
        self.audioproj = audioproj

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        ref_image_latents: torch.Tensor,
        face_emb: torch.Tensor,
        pose_vid: torch.Tensor,
        audio_emb: torch.Tensor,
        full_mask: torch.Tensor,
        face_mask: torch.Tensor,
        body_mask: torch.Tensor,
        motion_scale: torch.Tensor,
        uncond_img_fwd: bool = False,
        uncond_audio_fwd: bool = False,
    ):
        pose_cond_tensor = pose_vid.to(device="cuda")
        pose_fea = self.pose_guider(pose_cond_tensor)

        audio_emb = audio_emb.to(
            device=self.audioproj.device, dtype=self.audioproj.dtype)
        audio_emb = self.audioproj(audio_emb)   
        if not uncond_img_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=face_emb,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        if uncond_audio_fwd:
            # print("音频为无条件输入")
            audio_emb = torch.zeros_like(audio_emb) + audio_emb.mean() * 0  # 确保梯度计算

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            pose_cond_fea=pose_fea,            
            encoder_hidden_states=face_emb,
            audio_embedding=audio_emb,
            full_mask=full_mask,
            face_mask=face_mask,
            body_mask=body_mask,
            motion_scale = motion_scale
        ).sample

        return model_pred


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def log_validation(
    accelerator: Accelerator,
    vae: AutoencoderKL,
    image_enc,
    net: Net,
    scheduler: DDIMScheduler,
    width: int,
    height: int,
    clip_length: int = 25,
    generator: torch.Generator = None,
    cfg: dict = None,
    save_dir: str = None,
    global_step: int = 0,
    times: int = None,
) -> None:
    
    logger.info("Running validation... ")

    ori_net = accelerator.unwrap_model(net)
    reference_unet = ori_net.reference_unet
    denoising_unet = ori_net.denoising_unet
    pose_guider = ori_net.pose_guider
    audioproj = ori_net.audioproj

    if generator is None:
        generator = torch.manual_seed(42)
    tmp_denoising_unet = copy.deepcopy(denoising_unet)
    tmp_denoising_unet = tmp_denoising_unet.to(dtype=torch.float16)

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=tmp_denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to(accelerator.device)

    # test_cases = [
    #     (
    #         "./data/our_data/test/init_frames/chemistry#71087_slice1.png",
    #         "./data/our_data/test/videos_dwpose_gt/chemistry#71087_slice1.mp4",
    #     ),
    #     (
    #         "./data/our_data/test/init_frames/oliver#103786_slice0.png",
    #         "./data/our_data/test/videos_dwpose_gt/oliver#103786_slice0.mp4",
    #     ),
    # ]
    # image_processor = ImageProcessor((width, height))
    image_processor = ImageProcessor((width, height))
    audio_processor = AudioProcessor(
        cfg.data.sample_rate,
        cfg.data.fps,
        cfg.wav2vec_config.model_path,
        cfg.wav2vec_config.features == "last",
        os.path.dirname(cfg.audio_separator.model_path),
        os.path.basename(cfg.audio_separator.model_path),
        os.path.join(save_dir, '.cache', "audio_preprocess")
    )
    for idx, ref_img_path in enumerate(cfg.ref_img_path):
        audio_path = cfg.audio_path[idx]
        # source_image_pixels, \
        # clip_image_embeds, \
        # source_image_full_mask = image_processor.preprocess(
        #     ref_img_path, os.path.join(save_dir, '.cache'),
        #     config.face_expand_ratio)   
        

        audio_emb, audio_length = audio_processor.preprocess(
                                         audio_path, clip_length)
        audio_emb = process_audio_emb(audio_emb)
        audio_tensor = audio_emb[:clip_length]
        audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = audio_tensor.to(
            device=audioproj.device, dtype=audioproj.dtype)
        audio_tensor = audioproj(audio_tensor)



        # source_image_pixels = source_image_pixels.unsqueeze(0)        
        pose_vid = cfg.pose_vid_path
        face_mask_path = cfg.face_mask_path
        lips_mask_path = cfg.lips_mask_path
        face_mask_frames = read_frames(face_mask_path)
        lips_mask_frames = read_frames(lips_mask_path)
        pose_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )

        # ref_image_path, pose_video_path = test_case
        ref_name = Path(ref_img_path).stem
        pose_name = Path(pose_vid).stem
        ref_image_pil = Image.open(ref_img_path).convert("RGB")

        pose_list = []
        pose_tensor_list = []
        pose_images = read_frames(pose_vid)
        pose_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )

        for pose_image_pil in pose_images[:clip_length]:
            pose_tensor_list.append(pose_transform(pose_image_pil))
            pose_list.append(pose_image_pil)
        pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        pose_tensor = pose_tensor.transpose(0, 1)

        face_mask_list = []
        # TODO 获取当前片段的姿势序列
        for pose_image_pil in face_mask_frames[:clip_length]:
            face_mask_list.append(pose_image_pil)   

        lips_mask_list = []
        # TODO 获取当前片段的姿势序列
        for pose_image_pil in lips_mask_frames[:clip_length]:
            lips_mask_list.append(pose_image_pil)

        face_masks_list = []
        for face_mask_image in face_mask_list:
            # 转换为 NumPy 数组
            mask_array = np.array(face_mask_image)
            # 应用 blur_mask 函数
            processed_mask = blur_mask(mask_array, resize_dim=(64, 64), kernel_size=(31, 31))
            # 确保处理后的掩码是整数格式
            processed_mask = processed_mask.astype(np.uint8)
            # 转换回 PIL 图像并添加到列表中
            face_masks_list.append(Image.fromarray(processed_mask).convert("L"))   


        lips_masks_list = []
        for lips_mask_image in lips_mask_list:
            # 转换为 NumPy 数组
            mask_array = np.array(lips_mask_image)
            # 应用 blur_mask 函数
            processed_mask = blur_mask(mask_array, resize_dim=(64, 64), kernel_size=(21, 21))
            # 确保处理后的掩码是整数格式
            processed_mask = processed_mask.astype(np.uint8)
            # 转换回 PIL 图像并添加到列表中
            lips_masks_list.append(Image.fromarray(processed_mask).convert("L"))     
        pixel_values_face_mask, pixel_values_lips_mask = image_processor.preprocess_mov_mask(face_masks_list, 
                                                                                    lips_masks_list, config.face_expand_ratio, clip_length)

        source_image_full_mask = []
        for mask in pixel_values_face_mask:
            inverted_mask = 1.0 - mask.float()
            source_image_full_mask.append(inverted_mask)
        for i in range(len(pixel_values_lips_mask)):
            source_image_full_mask[i] = 1.0 + pixel_values_lips_mask[i].float()


        pipeline_output = pipe(
            ref_image=ref_image_pil,
            pose_images=pose_list,
            audio_tensor=audio_tensor,
            pixel_values_full_mask=source_image_full_mask,
            pixel_values_face_mask=pixel_values_face_mask,
            pixel_values_lip_mask=pixel_values_lips_mask,
            width=width,
            height=height,
            video_length=clip_length,
            num_inference_steps=config.inference_steps,
            guidance_scale=config.cfg_scale,
            generator=generator,
            motion_scale=config.motion_scale,
        )
        video = pipeline_output.videos
        tensor_result = video.squeeze(0)  # 移除批次维度
        tensor_result = tensor_result[:, :audio_length]  # 截取到音频的长度
        audio_name = os.path.basename(audio_path).split('.')[0]  # 获取音频文件的基本名称
        ref_name = os.path.basename(ref_img_path).split('.')[0]  # 获取参考图像的基本名称
        output_file = os.path.join(save_dir, f"{global_step}_{ref_name}_{audio_name}.mp4")  # 生成输出文件路径
        # 保存最终结果
        tensor_to_video(tensor_result, output_file, audio_path)  # 将张量转换为视频并保存到指定路径
        # Concat it with pose tensor
        # pose_tensor = pose_tensor.unsqueeze(0)
        # video = torch.cat([video, pose_tensor], dim=0)

    del tmp_denoising_unet
    del pipe
    del image_processor
    del audio_processor
    torch.cuda.empty_cache()


def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    module_dir = os.path.join(save_dir, "modules")
    if accelerator.is_main_process:
        init_output_dir([save_dir, checkpoint_dir, module_dir])
    if accelerator.is_main_process:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    validation_dir = os.path.join(save_dir, "validation")
    if accelerator.is_main_process:
        if not os.path.exists(validation_dir):
            os.makedirs(validation_dir)    

    inference_config_path = "./configs/inference/inference_v2.yaml"
    infer_config = OmegaConf.load(inference_config_path)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda", dtype=weight_dtype)

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        cfg.mm_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            infer_config.unet_additional_kwargs
        ),
    ).to(device="cuda")

    pose_guider = PoseGuider(
        conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)
    ).to(device="cuda", dtype=weight_dtype)

    stage1_ckpt_dir = cfg.stage1_ckpt_dir
    stage1_ckpt_step = cfg.stage1_ckpt_step
    denoising_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"denoising_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"reference_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    pose_guider.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"pose_guider-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )

    audioproj = AudioProjModel(
        seq_len=5,
        blocks=12,
        channels=768,
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
    ).to(device="cuda")

    # Freeze
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    pose_guider.requires_grad_(False)
    audioproj.requires_grad_(True)
    # Set motion module learnable
    # for name, module in denoising_unet.named_modules():
    #     if "motion_modules" in name:
    #         for params in module.parameters():
    #             params.requires_grad = True
    trainable_modules = cfg.trainable_para
    for name, module in denoising_unet.named_modules():
        if any(trainable_mod in name for trainable_mod in trainable_modules):
            for params in module.parameters():
                params.requires_grad_(True)


    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    net = Net(
        reference_unet,
        denoising_unet,
        pose_guider,
        reference_control_writer,
        reference_control_reader,
        audioproj,
    )

    # net = torch.nn.parallel.DistributedDataParallel(net, find_unused_parameters=True)

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
            * 0.5
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    logger.info(f"Total trainable params {len(trainable_params)}")
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )
    
    trainable_params_1 = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params_1:,}")
    param_size_bytes = trainable_params_1 * 4  # FP32 每个参数 4 字节
    param_size_gb = param_size_bytes / (1024 ** 3)
    print(f"Approx. model size (FP32): {param_size_gb:.2f} GB")

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    train_dataset = TalkingVideoDataset_move_mask(
        img_size=(cfg.data.train_width, cfg.data.train_height),
        sample_rate=cfg.data.sample_rate,
        n_sample_frames=cfg.data.n_sample_frames,
        n_motion_frames=cfg.data.n_motion_frames,
        audio_margin=cfg.data.audio_margin,
        data_meta_paths=cfg.data.meta_paths,
        wav2vec_cfg=cfg.wav2vec_config,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=4
    )

    # Prepare everything with our `accelerator`.
    (
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )
        # dump config file
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

    # Train!
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = checkpoint_dir
        # Get the most recent checkpoint
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        t_data_start = time.time()
        for step, batch in enumerate(train_dataloader):
            t_data = time.time() - t_data_start
            with accelerator.accumulate(net):
                # Convert videos to latent space
                pixel_values_vid = batch["pixel_values_vid"].to(weight_dtype)

                pixel_values_face_mask = batch["pixel_values_face_mask"]
                pixel_values_face_mask = get_attention_mask(
                    pixel_values_face_mask, weight_dtype
                )

                pixel_values_body_mask = batch["pixel_values_body_mask"]
                pixel_values_body_mask = get_attention_mask(
                    pixel_values_body_mask, weight_dtype
                )
                pixel_values_full_mask = batch["pixel_values_full_mask"]
                pixel_values_full_mask = get_attention_mask(
                    pixel_values_full_mask, weight_dtype
                )


                with torch.no_grad():
                    video_length = pixel_values_vid.shape[1]
                    pixel_values_vid = rearrange(
                        pixel_values_vid, "b f c h w -> (b f) c h w"
                    )
                    latents = vae.encode(pixel_values_vid).latent_dist.sample()
                    latents = rearrange(
                        latents, "(b f) c h w -> b c f h w", f=video_length
                    )
                    latents = latents * 0.18215

                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0:
                    noise += cfg.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1, 1),
                        device=latents.device,
                    )
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                pixel_values_pose = batch["pixel_values_pose"]  # (bs, f, c, H, W)
                pixel_values_pose = pixel_values_pose.transpose(
                    1, 2
                )  # (bs, c, f, H, W)

                uncond_fwd = random.random() < cfg.uncond_ratio
                uncond_audio_fwd = random.random() < cfg.uncond_audio_ratio

                clip_image_list = []
                ref_image_list = []
                for batch_idx, (ref_img, clip_img) in enumerate(
                    zip(
                        batch["pixel_values_ref_img"],
                        batch["clip_images"],
                    )
                ):
                    if uncond_fwd:
                        clip_image_list.append(torch.zeros_like(clip_img))
                    else:
                        clip_image_list.append(clip_img)
                    ref_image_list.append(ref_img)

                with torch.no_grad():
                    ref_img = torch.stack(ref_image_list, dim=0).to(
                        dtype=vae.dtype, device=vae.device
                    )
                    ref_img_and_motion = rearrange(
                        ref_img, "b f c h w -> (b f) c h w"
                    )
                    ref_image_latents = vae.encode(
                        ref_img_and_motion
                    ).latent_dist.sample()  # (bs, d, 64, 64)
                    ref_image_latents = ref_image_latents * 0.18215

                    clip_img = torch.stack(clip_image_list, dim=0).to(
                        dtype=image_enc.dtype, device=image_enc.device
                    )
                    clip_img = clip_img.to(device="cuda", dtype=weight_dtype)
                    clip_image_embeds = image_enc(
                        clip_img.to("cuda", dtype=weight_dtype)
                    ).image_embeds
                    clip_image_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)

                # add noise
                noisy_latents = train_noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                # Get the target for loss depending on the prediction type
                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )

                # ---- Forward!!! -----
                model_pred = net(
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    ref_image_latents=ref_image_latents,
                    face_emb=clip_image_embeds,
                    pose_vid = pixel_values_pose,                
                    full_mask=pixel_values_full_mask,
                    face_mask=pixel_values_face_mask,
                    body_mask=pixel_values_body_mask,
                    audio_emb=batch["audio_tensor"].to(
                        dtype=weight_dtype),
                    uncond_img_fwd=uncond_fwd,
                    uncond_audio_fwd=uncond_audio_fwd,
                    motion_scale = cfg.motion_scale
                )

                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps


                # Backpropagate
                accelerator.backward(loss)

                # for name, param in net.named_parameters():
                #     if param.requires_grad and param.grad is None:
                #         print(f"Parameter {name} is not used in forward pass")



                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                # if global_step % cfg.val.validation_steps == 0 or global_step==1:
                #     if accelerator.is_main_process:
                #         torch.cuda.empty_cache()
                #         generator = torch.Generator(device=accelerator.device)
                #         generator.manual_seed(cfg.seed)

                #         log_validation(
                #             accelerator=accelerator,
                #             vae=vae,
                #             image_enc=image_enc,                            
                #             net=net,
                #             scheduler=val_noise_scheduler,
                #             width=cfg.data.train_width,
                #             height=cfg.data.train_height,
                #             clip_length=32,
                #             cfg=cfg,
                #             save_dir=validation_dir,
                #             global_step=global_step,
                #             times=cfg.single_inference_times if cfg.single_inference_times is not None else None,
                #         )

                        # for sample_id, sample_dict in enumerate(sample_dicts):
                        #     sample_name = sample_dict["name"]
                        #     vid = sample_dict["vid"]
                        #     with TemporaryDirectory() as temp_dir:
                        #         out_file = Path(
                        #             f"{temp_dir}/{global_step:06d}-{sample_name}.gif"
                        #         )
                        #         save_videos_grid(vid, out_file, n_rows=2)
                        #         mlflow.log_artifact(out_file)

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "td": f"{t_data:.2f}s",
            }
            t_data_start = time.time()
            progress_bar.set_postfix(**logs)


        # save model after each epoch
            if accelerator.is_main_process:
                if (
                    global_step % cfg.checkpointing_steps == 0
                    or global_step == cfg.solver.max_train_steps
                ):
                    save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                    delete_additional_ckpt(save_dir, 5)
                    accelerator.save_state(save_path)
                    # save motion module only
                    unwrap_net = accelerator.unwrap_model(net)
                    save_checkpoint(
                        unwrap_net,
                        save_dir,
                        "net",
                        global_step,
                        total_limit=30
                    )

            if global_step >= cfg.solver.max_train_steps:
                break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


def save_checkpoint(model: torch.nn.Module, save_dir: str, prefix: str, ckpt_num: int, total_limit: int = -1) -> None:
    """
    Save the model's state_dict to a checkpoint file.

    If `total_limit` is provided, this function will remove the oldest checkpoints
    until the total number of checkpoints is less than the specified limit.

    Args:
        model (nn.Module): The model whose state_dict is to be saved.
        save_dir (str): The directory where the checkpoint will be saved.
        prefix (str): The prefix for the checkpoint file name.
        ckpt_num (int): The checkpoint number to be saved.
        total_limit (int, optional): The maximum number of checkpoints to keep.
            Defaults to None, in which case no checkpoints will be removed.

    Raises:
        FileNotFoundError: If the save directory does not exist.
        ValueError: If the checkpoint number is negative.
        OSError: If there is an error saving the checkpoint.
    """

    if not osp.exists(save_dir):
        raise FileNotFoundError(
            f"The save directory {save_dir} does not exist.")

    if ckpt_num < 0:
        raise ValueError(f"Checkpoint number {ckpt_num} must be non-negative.")

    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit > 0:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            print(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            print(
                f"Removing checkpoints: {', '.join(removing_checkpoints)}"
            )

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint_path = osp.join(
                    save_dir, removing_checkpoint)
                try:
                    os.remove(removing_checkpoint_path)
                except OSError as e:
                    print(
                        f"Error removing checkpoint {removing_checkpoint_path}: {e}")

    state_dict = model.state_dict()
    try:
        torch.save(state_dict, save_path)
        print(f"Checkpoint saved at {save_path}")
    except OSError as e:
        raise OSError(f"Error saving checkpoint at {save_path}: {e}") from e


def decode_latents(vae, latents):
    video_length = latents.shape[2]
    latents = 1 / 0.18215 * latents
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    # video = self.vae.decode(latents).sample
    video = []
    for frame_idx in tqdm(range(latents.shape[0])):
        video.append(vae.decode(latents[frame_idx : frame_idx + 1]).sample)
    video = torch.cat(video)
    video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
    video = (video / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    video = video.cpu().float().numpy()
    return video


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/stage2.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)
# nohup accelerate launch train_stage_2.py --config configs/train/stage2.yaml > multi_person_in_MMGT_for_response/stage2/train_.log 2>&1 &