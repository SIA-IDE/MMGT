# pylint: disable=R0801
"""
talking_video_dataset.py

This module defines the TalkingVideoDataset class, a custom PyTorch dataset 
for handling talking video data. The dataset uses video files, masks, and 
embeddings to prepare data for tasks such as video generation and 
speech-driven video animation.

Classes:
    TalkingVideoDataset

Dependencies:
    json
    random
    torch
    decord.VideoReader, decord.cpu
    PIL.Image
    torch.utils.data.Dataset
    torchvision.transforms

Example:
    from talking_video_dataset import TalkingVideoDataset
    from torch.utils.data import DataLoader

    # Example configuration for the Wav2Vec model
    class Wav2VecConfig:
        def __init__(self, audio_type, model_scale, features):
            self.audio_type = audio_type
            self.model_scale = model_scale
            self.features = features

    wav2vec_cfg = Wav2VecConfig(audio_type="wav2vec2", model_scale="base", features="feature")

    # Initialize dataset
    dataset = TalkingVideoDataset(
        img_size=(512, 512),
        sample_rate=16000,
        audio_margin=2,
        n_motion_frames=0,
        n_sample_frames=16,
        data_meta_paths=["path/to/meta1.json", "path/to/meta2.json"],
        wav2vec_cfg=wav2vec_cfg,
    )

    # Initialize dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Fetch one batch of data
    batch = next(iter(dataloader))
    print(batch["pixel_values_vid"].shape)  # Example output: (4, 16, 3, 512, 512)

The TalkingVideoDataset class provides methods for loading video frames, masks, 
audio embeddings, and other relevant data, applying transformations, and preparing 
the data for training and evaluation in a deep learning pipeline.

Attributes:
    img_size (tuple): The dimensions to resize the video frames to.
    sample_rate (int): The audio sample rate.
    audio_margin (int): The margin for audio sampling.
    n_motion_frames (int): The number of motion frames.
    n_sample_frames (int): The number of sample frames.
    data_meta_paths (list): List of paths to the JSON metadata files.
    wav2vec_cfg (object): Configuration for the Wav2Vec model.

Methods:
    augmentation(images, transform, state=None): Apply transformation to input images.
    __getitem__(index): Get a sample from the dataset at the specified index.
    __len__(): Return the length of the dataset.
"""

import json
import random
from typing import List
import cv2
import numpy as np
# import mxnet as mx
import torch
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPImageProcessor
# from hallo.utils.util improt 
from decord._ffi.base import DECORDError
import decord
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



def safe_video_reader(path):
    try:
        vr = VideoReader(path)
        # 尝试读取第一帧确认可用
        _ = vr[0]
        return vr
    except (DECORDError, RuntimeError, Exception) as e:
        print(f"[WARN] VideoReader 无法打开或解码: {path}，原因: {e}")
        return None




class TalkingVideoDataset_move_mask(Dataset):
    """
    A dataset class for processing talking video data.

    Args:
        img_size (tuple, optional): The size of the output images. Defaults to (512, 512).
        sample_rate (int, optional): The sample rate of the audio data. Defaults to 16000.
        audio_margin (int, optional): The margin for the audio data. Defaults to 2.
        n_motion_frames (int, optional): The number of motion frames. Defaults to 0.
        n_sample_frames (int, optional): The number of sample frames. Defaults to 16.
        data_meta_paths (list, optional): The paths to the data metadata. Defaults to None.
        wav2vec_cfg (dict, optional): The configuration for the wav2vec model. Defaults to None.

    Attributes:
        img_size (tuple): The size of the output images.
        sample_rate (int): The sample rate of the audio data.
        audio_margin (int): The margin for the audio data.
        n_motion_frames (int): The number of motion frames.
        n_sample_frames (int): The number of sample frames.
        data_meta_paths (list): The paths to the data metadata.
        wav2vec_cfg (dict): The configuration for the wav2vec model.
    """

    def __init__(
        self,
        img_size=(512, 512),
        sample_rate=16000,
        audio_margin=2,
        n_motion_frames=0,
        n_sample_frames=16,
        data_meta_paths=None,
        wav2vec_cfg=None,
        max_retry: int = 20
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.img_size = img_size
        self.audio_margin = audio_margin
        self.n_motion_frames = n_motion_frames
        self.n_sample_frames = n_sample_frames
        self.audio_type = wav2vec_cfg.audio_type
        self.audio_model = wav2vec_cfg.model_scale
        self.audio_features = wav2vec_cfg.features
        self.max_retry = max_retry
        self.clip_image_processor = CLIPImageProcessor()
        vid_meta = []
        for data_meta_path in data_meta_paths:
            with open(data_meta_path, "r", encoding="utf-8") as f:
                vid_meta.extend(json.load(f))
        self.vid_meta = vid_meta
        self.length = len(self.vid_meta)
        self.pixel_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_64 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 8, self.img_size[0] // 8)),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_32 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 16, self.img_size[0] // 16)),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_16 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 32, self.img_size[0] // 32)),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_8 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 64, self.img_size[0] // 64)),
                transforms.ToTensor(),
            ]
        )

    def augmentation(self, images, transform, state=None):
        """
        Apply the given transformation to the input images.
        
        Args:
            images (List[PIL.Image] or PIL.Image): The input images to be transformed.
            transform (torchvision.transforms.Compose): The transformation to be applied to the images.
            state (torch.ByteTensor, optional): The state of the random number generator. 
            If provided, it will set the RNG state to this value before applying the transformation. Defaults to None.

        Returns:
            torch.Tensor: The transformed images as a tensor. 
            If the input was a list of images, the tensor will have shape (f, c, h, w), 
            where f is the number of images, c is the number of channels, h is the height, and w is the width. 
            If the input was a single image, the tensor will have shape (c, h, w), 
            where c is the number of channels, h is the height, and w is the width.
        """
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):

        for _ in range(self.max_retry):
            video_meta = self.vid_meta[index]
            video_path = video_meta["video_path"]
            kps_path = video_meta["kps_path"]
            face_movment_mask = video_meta["face_mask_path"]
            lip_movement_mask = video_meta["lip_mask_path"]
            hands_movment_mask = video_meta["hands_mask_path"]

            # mask_path = video_meta["mask_path"]
            # TODO 改成文件夹，加载动态mask, 或者写一个函数，直接通过.npy文件获取mask
            # TODO 需要将现有的mask处理写成一个循环，然后循环处理，返回一个带有时间信息的mask
            # body_mask_union_path = video_meta.get("sep_mask_body", None)
            # face_mask_union_path = video_meta.get("sep_mask_face", None)
            # full_mask_union_path = video_meta.get("sep_mask_border", None)

            # face_emb_path = video_meta["face_emb_path"]


            audio_emb_path = video_meta[
                f"{self.audio_type}_emb_{self.audio_model}_{self.audio_features}"
            ]
            # tgt_mask_pil = Image.open(mask_path)
            # print(video_path)

            video_frames = safe_video_reader(video_path)
            kps_reader = safe_video_reader(kps_path)
            face_mask_frames = safe_video_reader(face_movment_mask)
            lip_mask_frames = safe_video_reader(lip_movement_mask)
            hands_mask_frames = safe_video_reader(hands_movment_mask)

            if None in (video_frames, kps_reader, face_mask_frames, lip_mask_frames, hands_mask_frames):
                return None
            
            assert len(video_frames) == len(
                kps_reader
            ), f"{len(video_frames) = } != {len(kps_reader) = } in {video_path}"        

            # assert tgt_mask_pil is not None, "Fail to load target mask."
            assert (video_frames is not None and len(video_frames) > 0), "Fail to load video frames."
            video_length = len(video_frames)

            min_len = self.n_sample_frames + self.n_motion_frames + 2 * self.audio_margin
            
            all_good = (
                None not in (video_frames, kps_reader,
                             face_mask_frames, lip_mask_frames, hands_mask_frames)
                and len(video_frames) == len(kps_reader)
                and len(video_frames) > min_len
            )
            if all_good:

                start_idx = random.randint(
                    self.n_motion_frames,
                    video_length - self.n_sample_frames - self.audio_margin - 1,
                )

                videos = video_frames[start_idx : start_idx + self.n_sample_frames]
                kpt_videos = kps_reader[start_idx : start_idx + self.n_sample_frames]
                face_mask_videos = face_mask_frames[start_idx : start_idx + self.n_sample_frames]
                lips_mask_videos = lip_mask_frames[start_idx : start_idx + self.n_sample_frames]
                background_mask_videos = hands_mask_frames[start_idx : start_idx + self.n_sample_frames]

                # face_mask_videos_np = face_mask_videos.asnumpy()
                # background_mask_videos_np = 255 - face_mask_videos_np
                # background_mask_videos = decord.ndarray.array(
                #     background_mask_videos_np, 
                #     ctx=decord.cpu(0)  # 如果想放到CPU上
                # )       
            # TODO 先准备好变化的mask视频，只mask脸部和手部，动态的进行mask，读取按照上述视频的分段方式进行分割
            # TODO 读取完之后转换为列表的形式，进行后续的处理
            # TODO 加上对kpath的处理

                frame_list = [
                    Image.fromarray(video).convert("RGB") for video in videos.asnumpy()
                ]

                pose_pil_image_list = [
                    Image.fromarray(video).convert("RGB") for video in kpt_videos.asnumpy()
                ]
            
                # TODO 牛逼直接乘了n份，那就好整了，直接把这个换成动态的就行了
                # 直接写一个函数
                face_mask_list = [
                    Image.fromarray(video).convert("RGB") for video in face_mask_videos.asnumpy()
                ]
                lips_mask_list = [
                    Image.fromarray(video).convert("RGB") for video in lips_mask_videos.asnumpy()
                ]
                background_mask_list = [
                    Image.fromarray(video).convert("RGB") for video in background_mask_videos.asnumpy()
                ]
                # face_masks_list = [Image.open(face_mask_union_path)] * self.n_sample_frames
                # body_masks_list = [Image.open(body_mask_union_path)] * self.n_sample_frames
                # full_masks_list = [Image.open(full_mask_union_path)] * self.n_sample_frames

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
                    
                # full_masks_list = [Image.open(full_mask_union_path)] * self.n_sample_frames
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


                # 逐帧处理 lips_masks_list
                background_masks_list = []
                for background_mask_image in background_mask_list:
                    # 转换为 NumPy 数组
                    mask_array = np.array(background_mask_image)
                    # 应用 blur_mask 函数
                    processed_mask = blur_mask(mask_array, resize_dim=(64, 64), kernel_size=(21, 21))
                    # 确保处理后的掩码是整数格式
                    processed_mask = processed_mask.astype(np.uint8)
                    # 转换回 PIL 图像并添加到列表中
                    background_masks_list.append(Image.fromarray(processed_mask).convert("L"))      


                assert face_masks_list[0] is not None, "Fail to load face mask."
                assert lips_masks_list[0] is not None, "Fail to load lip mask."
                assert background_masks_list[0] is not None, "Fail to load lip mask."

                # assert full_masks_list[0] is not None, "Fail to load full mask."


                audio_emb = torch.load(audio_emb_path)
                indices = (
                    torch.arange(2 * self.audio_margin + 1) - self.audio_margin
                )  # Generates [-2, -1, 0, 1, 2]
                center_indices = torch.arange(
                    start_idx,
                    start_idx + self.n_sample_frames,
                ).unsqueeze(1) + indices.unsqueeze(0)
                audio_tensor = audio_emb[center_indices]

                ref_img_idx = random.randint(
                    self.n_motion_frames,
                    video_length - self.n_sample_frames - self.audio_margin - 1,
                )
                
                ref_img = video_frames[ref_img_idx].asnumpy()
                ref_img_pil = Image.fromarray(ref_img)

                # ref_img_pil = Image.open(video_frames[ref_img_idx])

                if self.n_motion_frames > 0:
                    motions = video_frames[start_idx - self.n_motion_frames : start_idx]
                    motion_list = [
                        Image.fromarray(motion).convert("RGB") for motion in motions.asnumpy()
                    ]

                # transform
                state = torch.get_rng_state()
                pixel_values_vid = self.augmentation(frame_list, self.pixel_transform, state)
                pixel_values_pose = self.augmentation(
                    pose_pil_image_list, self.cond_transform, state
                )        

                # pixel_values_mask = self.augmentation(tgt_mask_pil, self.cond_transform, state)
                # pixel_values_mask = pixel_values_mask.repeat(3, 1, 1)

                # TODO 将mask转换成了不同的尺寸
                pixel_values_face_mask = [
                    self.augmentation(face_masks_list, self.attn_transform_64, state),
                    self.augmentation(face_masks_list, self.attn_transform_32, state),
                    self.augmentation(face_masks_list, self.attn_transform_16, state),
                    self.augmentation(face_masks_list, self.attn_transform_8, state),
                ]
                pixel_values_body_mask = [
                    self.augmentation(lips_masks_list, self.attn_transform_64, state),
                    self.augmentation(lips_masks_list, self.attn_transform_32, state),
                    self.augmentation(lips_masks_list, self.attn_transform_16, state),
                    self.augmentation(lips_masks_list, self.attn_transform_8, state),
                ]

                pixel_values_full_mask = [
                    self.augmentation(background_masks_list, self.attn_transform_64, state),
                    self.augmentation(background_masks_list, self.attn_transform_32, state),
                    self.augmentation(background_masks_list, self.attn_transform_16, state),
                    self.augmentation(background_masks_list, self.attn_transform_8, state),
                ]

                pixel_values_ref_img = self.augmentation(ref_img_pil, self.pixel_transform, state)
                pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)
                
                # if self.n_motion_frames > 0:
                #     pixel_values_motion = self.augmentation(
                #         motion_list, self.pixel_transform, state
                #     )
                #     pixel_values_ref_img = torch.cat(
                #         [pixel_values_ref_img, pixel_values_motion], dim=0
                #     )

                clip_image = self.clip_image_processor(
                    images=ref_img_pil, return_tensors="pt"
                ).pixel_values[0]  

                sample = {
                    "video_dir": video_path,
                    "pixel_values_vid": pixel_values_vid,
                    "pixel_values_pose": pixel_values_pose,
                    # "pixel_values_mask": pixel_values_mask,
                    "pixel_values_face_mask": pixel_values_face_mask,
                    "pixel_values_body_mask": pixel_values_body_mask,
                    "pixel_values_full_mask": pixel_values_full_mask,
                    "audio_tensor": audio_tensor,
                    "pixel_values_ref_img": pixel_values_ref_img,
                    "clip_images": clip_image,
                }
                return sample
        
            index = random.randrange(len(self.vid_meta))

        raise RuntimeError(
            f"Failed to fetch a valid sample (> {self.min_len} frames) "
            f"after {self.max_retry} retries. "
            f"Please check dataset integrity."
        )        

    def __len__(self):
        return len(self.vid_meta)

class TalkingVideoDataset_move_mas_pats(Dataset):
    """
    A dataset class for processing talking video data.

    Args:
        img_size (tuple, optional): The size of the output images. Defaults to (512, 512).
        sample_rate (int, optional): The sample rate of the audio data. Defaults to 16000.
        audio_margin (int, optional): The margin for the audio data. Defaults to 2.
        n_motion_frames (int, optional): The number of motion frames. Defaults to 0.
        n_sample_frames (int, optional): The number of sample frames. Defaults to 16.
        data_meta_paths (list, optional): The paths to the data metadata. Defaults to None.
        wav2vec_cfg (dict, optional): The configuration for the wav2vec model. Defaults to None.

    Attributes:
        img_size (tuple): The size of the output images.
        sample_rate (int): The sample rate of the audio data.
        audio_margin (int): The margin for the audio data.
        n_motion_frames (int): The number of motion frames.
        n_sample_frames (int): The number of sample frames.
        data_meta_paths (list): The paths to the data metadata.
        wav2vec_cfg (dict): The configuration for the wav2vec model.
    """

    def __init__(
        self,
        img_size=(512, 512),
        sample_rate=16000,
        audio_margin=2,
        n_motion_frames=0,
        n_sample_frames=16,
        data_meta_paths=None,
        wav2vec_cfg=None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.img_size = img_size
        self.audio_margin = audio_margin
        self.n_motion_frames = n_motion_frames
        self.n_sample_frames = n_sample_frames
        self.audio_type = wav2vec_cfg.audio_type
        self.audio_model = wav2vec_cfg.model_scale
        self.audio_features = wav2vec_cfg.features
        self.clip_image_processor = CLIPImageProcessor()
        vid_meta = []
        for data_meta_path in data_meta_paths:
            with open(data_meta_path, "r", encoding="utf-8") as f:
                vid_meta.extend(json.load(f))
        self.vid_meta = vid_meta
        self.length = len(self.vid_meta)
        self.pixel_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_64 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 8, self.img_size[0] // 8)),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_32 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 16, self.img_size[0] // 16)),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_16 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 32, self.img_size[0] // 32)),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_8 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 64, self.img_size[0] // 64)),
                transforms.ToTensor(),
            ]
        )

    def augmentation(self, images, transform, state=None):
        """
        Apply the given transformation to the input images.
        
        Args:
            images (List[PIL.Image] or PIL.Image): The input images to be transformed.
            transform (torchvision.transforms.Compose): The transformation to be applied to the images.
            state (torch.ByteTensor, optional): The state of the random number generator. 
            If provided, it will set the RNG state to this value before applying the transformation. Defaults to None.

        Returns:
            torch.Tensor: The transformed images as a tensor. 
            If the input was a list of images, the tensor will have shape (f, c, h, w), 
            where f is the number of images, c is the number of channels, h is the height, and w is the width. 
            If the input was a single image, the tensor will have shape (c, h, w), 
            where c is the number of channels, h is the height, and w is the width.
        """
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]
        face_movment_mask = video_meta["face_mask_path"]
        lip_movement_mask = video_meta["lip_mask_path"]

        # mask_path = video_meta["mask_path"]
        # TODO 改成文件夹，加载动态mask, 或者写一个函数，直接通过.npy文件获取mask
        # TODO 需要将现有的mask处理写成一个循环，然后循环处理，返回一个带有时间信息的mask
        # body_mask_union_path = video_meta.get("sep_mask_body", None)
        # face_mask_union_path = video_meta.get("sep_mask_face", None)
        # full_mask_union_path = video_meta.get("sep_mask_border", None)

        # face_emb_path = video_meta["face_emb_path"]


        audio_emb_path = video_meta[
            f"{self.audio_type}_emb_{self.audio_model}_{self.audio_features}"
        ]
        # tgt_mask_pil = Image.open(mask_path)

        video_frames = VideoReader(video_path, ctx=cpu(0))
        kps_reader = VideoReader(kps_path)
        face_mask_frames = VideoReader(face_movment_mask)
        lip_mask_frames = VideoReader(lip_movement_mask)

        assert len(video_frames) == len(
            kps_reader
        ), f"{len(video_frames) = } != {len(kps_reader) = } in {video_path}"        

        # assert tgt_mask_pil is not None, "Fail to load target mask."
        assert (video_frames is not None and len(video_frames) > 0), "Fail to load video frames."
        video_length = len(video_frames)

        assert (
            video_length
            > self.n_sample_frames + self.n_motion_frames + 2 * self.audio_margin
        )

        start_idx = random.randint(
            self.n_motion_frames,
            video_length - self.n_sample_frames - self.audio_margin - 1,
        )

        videos = video_frames[start_idx : start_idx + self.n_sample_frames]
        kpt_videos = kps_reader[start_idx : start_idx + self.n_sample_frames]
        face_mask_videos = face_mask_frames[start_idx : start_idx + self.n_sample_frames]
        lips_mask_videos = lip_mask_frames[start_idx : start_idx + self.n_sample_frames]
        lips_mask_videos_np = lips_mask_videos.asnumpy()
        face_mask_videos_np = face_mask_videos.asnumpy()
        background_mask_videos_np = 255 - face_mask_videos_np + lips_mask_videos_np
        background_mask_videos = decord.ndarray.array(
            background_mask_videos_np, 
            ctx=decord.cpu(0)  # 如果想放到CPU上
        )       
       # TODO 先准备好变化的mask视频，只mask脸部和手部，动态的进行mask，读取按照上述视频的分段方式进行分割
       # TODO 读取完之后转换为列表的形式，进行后续的处理
       # TODO 加上对kpath的处理

        frame_list = [
            Image.fromarray(video).convert("RGB") for video in videos.asnumpy()
        ]

        pose_pil_image_list = [
            Image.fromarray(video).convert("RGB") for video in kpt_videos.asnumpy()
        ]
      
        # TODO 牛逼直接乘了n份，那就好整了，直接把这个换成动态的就行了
        # 直接写一个函数
        face_mask_list = [
            Image.fromarray(video).convert("RGB") for video in face_mask_videos.asnumpy()
        ]
        lips_mask_list = [
            Image.fromarray(video).convert("RGB") for video in lips_mask_videos.asnumpy()
        ]
        background_mask_list = [
            Image.fromarray(video).convert("RGB") for video in background_mask_videos.asnumpy()
        ]
        # face_masks_list = [Image.open(face_mask_union_path)] * self.n_sample_frames
        # body_masks_list = [Image.open(body_mask_union_path)] * self.n_sample_frames
        # full_masks_list = [Image.open(full_mask_union_path)] * self.n_sample_frames

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
            
        # full_masks_list = [Image.open(full_mask_union_path)] * self.n_sample_frames
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


        # 逐帧处理 lips_masks_list
        background_masks_list = []
        for background_mask_image in background_mask_list:
            # 转换为 NumPy 数组
            mask_array = np.array(background_mask_image)
            # 应用 blur_mask 函数
            processed_mask = blur_mask(mask_array, resize_dim=(64, 64), kernel_size=(21, 21))
            # 确保处理后的掩码是整数格式
            processed_mask = processed_mask.astype(np.uint8)
            # 转换回 PIL 图像并添加到列表中
            background_masks_list.append(Image.fromarray(processed_mask).convert("L"))      


        assert face_masks_list[0] is not None, "Fail to load face mask."
        assert lips_masks_list[0] is not None, "Fail to load lip mask."
        assert background_masks_list[0] is not None, "Fail to load lip mask."

        # assert full_masks_list[0] is not None, "Fail to load full mask."


        audio_emb = torch.load(audio_emb_path)
        indices = (
            torch.arange(2 * self.audio_margin + 1) - self.audio_margin
        )  # Generates [-2, -1, 0, 1, 2]
        center_indices = torch.arange(
            start_idx,
            start_idx + self.n_sample_frames,
        ).unsqueeze(1) + indices.unsqueeze(0)
        audio_tensor = audio_emb[center_indices]

        ref_img_idx = random.randint(
            self.n_motion_frames,
            video_length - self.n_sample_frames - self.audio_margin - 1,
        )
        
        ref_img = video_frames[ref_img_idx].asnumpy()
        ref_img_pil = Image.fromarray(ref_img)

        # ref_img_pil = Image.open(video_frames[ref_img_idx])

        if self.n_motion_frames > 0:
            motions = video_frames[start_idx - self.n_motion_frames : start_idx]
            motion_list = [
                Image.fromarray(motion).convert("RGB") for motion in motions.asnumpy()
            ]

        # transform
        state = torch.get_rng_state()
        pixel_values_vid = self.augmentation(frame_list, self.pixel_transform, state)
        pixel_values_pose = self.augmentation(
            pose_pil_image_list, self.cond_transform, state
        )        

        # pixel_values_mask = self.augmentation(tgt_mask_pil, self.cond_transform, state)
        # pixel_values_mask = pixel_values_mask.repeat(3, 1, 1)

        # TODO 将mask转换成了不同的尺寸
        pixel_values_face_mask = [
            self.augmentation(face_masks_list, self.attn_transform_64, state),
            self.augmentation(face_masks_list, self.attn_transform_32, state),
            self.augmentation(face_masks_list, self.attn_transform_16, state),
            self.augmentation(face_masks_list, self.attn_transform_8, state),
        ]
        pixel_values_body_mask = [
            self.augmentation(lips_masks_list, self.attn_transform_64, state),
            self.augmentation(lips_masks_list, self.attn_transform_32, state),
            self.augmentation(lips_masks_list, self.attn_transform_16, state),
            self.augmentation(lips_masks_list, self.attn_transform_8, state),
        ]

        pixel_values_full_mask = [
            self.augmentation(background_masks_list, self.attn_transform_64, state),
            self.augmentation(background_masks_list, self.attn_transform_32, state),
            self.augmentation(background_masks_list, self.attn_transform_16, state),
            self.augmentation(background_masks_list, self.attn_transform_8, state),
        ]

        pixel_values_ref_img = self.augmentation(ref_img_pil, self.pixel_transform, state)
        pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)
           
        # if self.n_motion_frames > 0:
        #     pixel_values_motion = self.augmentation(
        #         motion_list, self.pixel_transform, state
        #     )
        #     pixel_values_ref_img = torch.cat(
        #         [pixel_values_ref_img, pixel_values_motion], dim=0
        #     )

        clip_image = self.clip_image_processor(
            images=ref_img_pil, return_tensors="pt"
        ).pixel_values[0]  

        sample = {
            "video_dir": video_path,
            "pixel_values_vid": pixel_values_vid,
            "pixel_values_pose": pixel_values_pose,
            # "pixel_values_mask": pixel_values_mask,
            "pixel_values_face_mask": pixel_values_face_mask,
            "pixel_values_body_mask": pixel_values_body_mask,
            "pixel_values_full_mask": pixel_values_full_mask,
            "audio_tensor": audio_tensor,
            "pixel_values_ref_img": pixel_values_ref_img,
            "clip_images": clip_image,
        }

        return sample

    def __len__(self):
        return len(self.vid_meta)

# pylint: disable=R0801
"""
talking_video_dataset.py

This module defines the TalkingVideoDataset class, a custom PyTorch dataset 
for handling talking video data. The dataset uses video files, masks, and 
embeddings to prepare data for tasks such as video generation and 
speech-driven video animation.

Classes:
    TalkingVideoDataset

Dependencies:
    json
    random
    torch
    decord.VideoReader, decord.cpu
    PIL.Image
    torch.utils.data.Dataset
    torchvision.transforms

Example:
    from talking_video_dataset import TalkingVideoDataset
    from torch.utils.data import DataLoader

    # Example configuration for the Wav2Vec model
    class Wav2VecConfig:
        def __init__(self, audio_type, model_scale, features):
            self.audio_type = audio_type
            self.model_scale = model_scale
            self.features = features

    wav2vec_cfg = Wav2VecConfig(audio_type="wav2vec2", model_scale="base", features="feature")

    # Initialize dataset
    dataset = TalkingVideoDataset(
        img_size=(512, 512),
        sample_rate=16000,
        audio_margin=2,
        n_motion_frames=0,
        n_sample_frames=16,
        data_meta_paths=["path/to/meta1.json", "path/to/meta2.json"],
        wav2vec_cfg=wav2vec_cfg,
    )

    # Initialize dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Fetch one batch of data
    batch = next(iter(dataloader))
    print(batch["pixel_values_vid"].shape)  # Example output: (4, 16, 3, 512, 512)

The TalkingVideoDataset class provides methods for loading video frames, masks, 
audio embeddings, and other relevant data, applying transformations, and preparing 
the data for training and evaluation in a deep learning pipeline.

Attributes:
    img_size (tuple): The dimensions to resize the video frames to.
    sample_rate (int): The audio sample rate.
    audio_margin (int): The margin for audio sampling.
    n_motion_frames (int): The number of motion frames.
    n_sample_frames (int): The number of sample frames.
    data_meta_paths (list): List of paths to the JSON metadata files.
    wav2vec_cfg (object): Configuration for the Wav2Vec model.

Methods:
    augmentation(images, transform, state=None): Apply transformation to input images.
    __getitem__(index): Get a sample from the dataset at the specified index.
    __len__(): Return the length of the dataset.
"""

import json
import random
from typing import List
import cv2
import numpy as np
# import mxnet as mx
import torch
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPImageProcessor
# from hallo.utils.util improt 
import decord
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








class TalkingVideoDataset_move_mask_no_audio(Dataset):
    """
    A dataset class for processing talking video data.

    Args:
        img_size (tuple, optional): The size of the output images. Defaults to (512, 512).
        sample_rate (int, optional): The sample rate of the audio data. Defaults to 16000.
        audio_margin (int, optional): The margin for the audio data. Defaults to 2.
        n_motion_frames (int, optional): The number of motion frames. Defaults to 0.
        n_sample_frames (int, optional): The number of sample frames. Defaults to 16.
        data_meta_paths (list, optional): The paths to the data metadata. Defaults to None.
        wav2vec_cfg (dict, optional): The configuration for the wav2vec model. Defaults to None.

    Attributes:
        img_size (tuple): The size of the output images.
        sample_rate (int): The sample rate of the audio data.
        audio_margin (int): The margin for the audio data.
        n_motion_frames (int): The number of motion frames.
        n_sample_frames (int): The number of sample frames.
        data_meta_paths (list): The paths to the data metadata.
        wav2vec_cfg (dict): The configuration for the wav2vec model.
    """

    def __init__(
        self,
        img_size=(512, 512),
        sample_rate=16000,
        audio_margin=2,
        n_motion_frames=0,
        n_sample_frames=16,
        data_meta_paths=None,
        wav2vec_cfg=None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.img_size = img_size
        self.audio_margin = audio_margin
        self.n_motion_frames = n_motion_frames
        self.n_sample_frames = n_sample_frames
        self.audio_type = wav2vec_cfg.audio_type
        self.audio_model = wav2vec_cfg.model_scale
        self.audio_features = wav2vec_cfg.features
        self.clip_image_processor = CLIPImageProcessor()
        vid_meta = []
        for data_meta_path in data_meta_paths:
            with open(data_meta_path, "r", encoding="utf-8") as f:
                vid_meta.extend(json.load(f))
        self.vid_meta = vid_meta
        self.length = len(self.vid_meta)
        self.pixel_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_64 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 8, self.img_size[0] // 8)),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_32 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 16, self.img_size[0] // 16)),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_16 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 32, self.img_size[0] // 32)),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_8 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 64, self.img_size[0] // 64)),
                transforms.ToTensor(),
            ]
        )

    def augmentation(self, images, transform, state=None):
        """
        Apply the given transformation to the input images.
        
        Args:
            images (List[PIL.Image] or PIL.Image): The input images to be transformed.
            transform (torchvision.transforms.Compose): The transformation to be applied to the images.
            state (torch.ByteTensor, optional): The state of the random number generator. 
            If provided, it will set the RNG state to this value before applying the transformation. Defaults to None.

        Returns:
            torch.Tensor: The transformed images as a tensor. 
            If the input was a list of images, the tensor will have shape (f, c, h, w), 
            where f is the number of images, c is the number of channels, h is the height, and w is the width. 
            If the input was a single image, the tensor will have shape (c, h, w), 
            where c is the number of channels, h is the height, and w is the width.
        """
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]
        face_movment_mask = video_meta["face_mask_path"]
        lip_movement_mask = video_meta["lip_mask_path"]

        # mask_path = video_meta["mask_path"]
        # TODO 改成文件夹，加载动态mask, 或者写一个函数，直接通过.npy文件获取mask
        # TODO 需要将现有的mask处理写成一个循环，然后循环处理，返回一个带有时间信息的mask
        # body_mask_union_path = video_meta.get("sep_mask_body", None)
        # face_mask_union_path = video_meta.get("sep_mask_face", None)
        # full_mask_union_path = video_meta.get("sep_mask_border", None)

        # face_emb_path = video_meta["face_emb_path"]


        audio_emb_path = video_meta[
            f"{self.audio_type}_emb_{self.audio_model}_{self.audio_features}"
        ]
        # tgt_mask_pil = Image.open(mask_path)

        video_frames = VideoReader(video_path, ctx=cpu(0))
        kps_reader = VideoReader(kps_path)
        face_mask_frames = VideoReader(face_movment_mask)
        lip_mask_frames = VideoReader(lip_movement_mask)

        assert len(video_frames) == len(
            kps_reader
        ), f"{len(video_frames) = } != {len(kps_reader) = } in {video_path}"        

        # assert tgt_mask_pil is not None, "Fail to load target mask."
        assert (video_frames is not None and len(video_frames) > 0), "Fail to load video frames."
        video_length = len(video_frames)

        assert (
            video_length
            > self.n_sample_frames + self.n_motion_frames + 2 * self.audio_margin
        )

        start_idx = random.randint(
            self.n_motion_frames,
            video_length - self.n_sample_frames - self.audio_margin - 1,
        )

        videos = video_frames[start_idx : start_idx + self.n_sample_frames]
        kpt_videos = kps_reader[start_idx : start_idx + self.n_sample_frames]
        face_mask_videos = face_mask_frames[start_idx : start_idx + self.n_sample_frames]
        lips_mask_videos = lip_mask_frames[start_idx : start_idx + self.n_sample_frames]
        lips_mask_videos_np = lips_mask_videos.asnumpy()
        face_mask_videos_np = face_mask_videos.asnumpy()
        background_mask_videos_np = 255 - face_mask_videos_np + lips_mask_videos_np
        background_mask_videos = decord.ndarray.array(
            background_mask_videos_np, 
            ctx=decord.cpu(0)  # 如果想放到CPU上
        )       
       # TODO 先准备好变化的mask视频，只mask脸部和手部，动态的进行mask，读取按照上述视频的分段方式进行分割
       # TODO 读取完之后转换为列表的形式，进行后续的处理
       # TODO 加上对kpath的处理

        frame_list = [
            Image.fromarray(video).convert("RGB") for video in videos.asnumpy()
        ]

        pose_pil_image_list = [
            Image.fromarray(video).convert("RGB") for video in kpt_videos.asnumpy()
        ]
      
        # TODO 牛逼直接乘了n份，那就好整了，直接把这个换成动态的就行了
        # 直接写一个函数
        face_mask_list = [
            Image.fromarray(video).convert("RGB") for video in face_mask_videos.asnumpy()
        ]
        lips_mask_list = [
            Image.fromarray(video).convert("RGB") for video in lips_mask_videos.asnumpy()
        ]
        background_mask_list = [
            Image.fromarray(video).convert("RGB") for video in background_mask_videos.asnumpy()
        ]
        # face_masks_list = [Image.open(face_mask_union_path)] * self.n_sample_frames
        # body_masks_list = [Image.open(body_mask_union_path)] * self.n_sample_frames
        # full_masks_list = [Image.open(full_mask_union_path)] * self.n_sample_frames

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
            
        # full_masks_list = [Image.open(full_mask_union_path)] * self.n_sample_frames
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


        # 逐帧处理 lips_masks_list
        background_masks_list = []
        for background_mask_image in background_mask_list:
            # 转换为 NumPy 数组
            mask_array = np.array(background_mask_image)
            # 应用 blur_mask 函数
            processed_mask = blur_mask(mask_array, resize_dim=(64, 64), kernel_size=(21, 21))
            # 确保处理后的掩码是整数格式
            processed_mask = processed_mask.astype(np.uint8)
            # 转换回 PIL 图像并添加到列表中
            background_masks_list.append(Image.fromarray(processed_mask).convert("L"))      


        assert face_masks_list[0] is not None, "Fail to load face mask."
        assert lips_masks_list[0] is not None, "Fail to load lip mask."
        assert background_masks_list[0] is not None, "Fail to load lip mask."

        # assert full_masks_list[0] is not None, "Fail to load full mask."


        audio_emb = torch.load(audio_emb_path)
        indices = (
            torch.arange(2 * self.audio_margin + 1) - self.audio_margin
        )  # Generates [-2, -1, 0, 1, 2]
        center_indices = torch.arange(
            start_idx,
            start_idx + self.n_sample_frames,
        ).unsqueeze(1) + indices.unsqueeze(0)
        audio_tensor = audio_emb[center_indices]

        ref_img_idx = random.randint(
            self.n_motion_frames,
            video_length - self.n_sample_frames - self.audio_margin - 1,
        )
        
        ref_img = video_frames[ref_img_idx].asnumpy()
        ref_img_pil = Image.fromarray(ref_img)

        # ref_img_pil = Image.open(video_frames[ref_img_idx])

        if self.n_motion_frames > 0:
            motions = video_frames[start_idx - self.n_motion_frames : start_idx]
            motion_list = [
                Image.fromarray(motion).convert("RGB") for motion in motions.asnumpy()
            ]

        # transform
        state = torch.get_rng_state()
        pixel_values_vid = self.augmentation(frame_list, self.pixel_transform, state)
        pixel_values_pose = self.augmentation(
            pose_pil_image_list, self.cond_transform, state
        )        

        # pixel_values_mask = self.augmentation(tgt_mask_pil, self.cond_transform, state)
        # pixel_values_mask = pixel_values_mask.repeat(3, 1, 1)

        # TODO 将mask转换成了不同的尺寸
        pixel_values_face_mask = [
            self.augmentation(face_masks_list, self.attn_transform_64, state),
            self.augmentation(face_masks_list, self.attn_transform_32, state),
            self.augmentation(face_masks_list, self.attn_transform_16, state),
            self.augmentation(face_masks_list, self.attn_transform_8, state),
        ]
        pixel_values_body_mask = [
            self.augmentation(lips_masks_list, self.attn_transform_64, state),
            self.augmentation(lips_masks_list, self.attn_transform_32, state),
            self.augmentation(lips_masks_list, self.attn_transform_16, state),
            self.augmentation(lips_masks_list, self.attn_transform_8, state),
        ]

        pixel_values_full_mask = [
            self.augmentation(background_masks_list, self.attn_transform_64, state),
            self.augmentation(background_masks_list, self.attn_transform_32, state),
            self.augmentation(background_masks_list, self.attn_transform_16, state),
            self.augmentation(background_masks_list, self.attn_transform_8, state),
        ]

        pixel_values_ref_img = self.augmentation(ref_img_pil, self.pixel_transform, state)
        pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)
           
        if self.n_motion_frames > 0:
            pixel_values_motion = self.augmentation(
                motion_list, self.pixel_transform, state
            )
            pixel_values_ref_img = torch.cat(
                [pixel_values_ref_img, pixel_values_motion], dim=0
            )

        clip_image = self.clip_image_processor(
            images=ref_img_pil, return_tensors="pt"
        ).pixel_values[0]  

        sample = {
            "video_dir": video_path,
            "pixel_values_vid": pixel_values_vid,
            "pixel_values_pose": pixel_values_pose,
            # "pixel_values_mask": pixel_values_mask,
            "pixel_values_face_mask": pixel_values_face_mask,
            "pixel_values_body_mask": pixel_values_body_mask,
            "pixel_values_full_mask": pixel_values_full_mask,
            "audio_tensor": audio_tensor,
            "pixel_values_ref_img": pixel_values_ref_img,
            "clip_images": clip_image,
        }

        return sample

    def __len__(self):
        return len(self.vid_meta)