# pylint: disable=W0718
"""
This module is responsible for processing images, particularly for face-related tasks.
It uses various libraries such as OpenCV, NumPy, and InsightFace to perform tasks like
face detection, augmentation, and mask rendering. The ImageProcessor class encapsulates
the functionality for these operations.
"""
import os
from typing import List

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
# from mediapipe.tasks import BaseOptions, VisionRunningMode
# from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, bodyLandmarker, bodyLandmarkerOptions

from ..utils.util import blur_mask

from ..utils.util_2 import (process_images_in_folder, process_single_images)
from transformers import CLIPVisionModelWithProjection
from transformers import CLIPImageProcessor

MEAN = 0.5
STD = 0.5

class ImageProcessor:
    """
    ImageProcessor is a class responsible for processing images, particularly for face-related tasks.
    It takes in an image and performs various operations such as augmentation, face detection,
    face embedding extraction, and rendering a face mask. The processed images are then used for
    further analysis or recognition purposes.

    Attributes:
        img_size (int): The size of the image to be processed.
        face_analysis_model_path (str): The path to the face analysis model.

    Methods:
        preprocess(source_image_path, cache_dir):
            Preprocesses the input image by performing augmentation, face detection,
            face embedding extraction, and rendering a face mask.

        close():
            Closes the ImageProcessor and releases any resources being used.

        _augmentation(images, transform, state=None):
            Applies image augmentation to the input images using the given transform and state.

        __enter__():
            Enters a runtime context and returns the ImageProcessor object.

        __exit__(_exc_type, _exc_val, _exc_tb):
            Exits a runtime context and bodyles any exceptions that occurred during the processing.
    """
    def __init__(self, img_size) -> None:
        self.img_size = img_size

        self.pixel_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([MEAN], [STD]),
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

        self.clip_image_processor = CLIPImageProcessor()

    def preprocess(self, source_image_path: str, cache_dir: str, face_region_ratio: float):
        """
        Apply preprocessing to the source image to prepare for face analysis.

        Parameters:
            source_image_path (str): The path to the source image.
            cache_dir (str): The directory to cache intermediate results.

        Returns:
            None
        """
        image_encoder_path = './pretrained_models/sd-image-variations-diffusers/image_encoder'
        image_enc = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path,
        ).to(dtype=torch.float16, device="cuda")

        ref_image_pil = Image.open(source_image_path).convert("RGB")
        pixel_values_ref_img = self._augmentation(ref_image_pil, self.pixel_transform)
        # 1. image augmentation
        clip_image = self.clip_image_processor(
            images=ref_image_pil, return_tensors="pt"
        ).pixel_values[0] 

        clip_image = clip_image.unsqueeze(0)
        clip_image_embeds = image_enc(
            clip_image.to("cuda", dtype=torch.float16)
        ).image_embeds

        # # 使用yolov5生成mask
        # ################################################################
        # face_mask, body_mask = process_single_images(source_image_path)
        # # 4 gaussian blur
        # blur_face_mask = blur_mask(face_mask, (64, 64), (41, 41))
        # blur_body_mask = blur_mask(body_mask, (64, 64), (71, 71))

        # # 5 seperate mask 
        # sep_face_mask = blur_face_mask
        # sep_body_mask = cv2.subtract(blur_body_mask, sep_face_mask)
        # sep_pose_mask = 255.0 - blur_face_mask

        # sep_face_mask = sep_face_mask.astype(np.uint8)
        # sep_body_mask = sep_body_mask.astype(np.uint8)
        # sep_pose_mask = sep_pose_mask.astype(np.uint8)

        sep_pose_mask = np.zeros((64, 64), dtype=np.uint8)
        # 将mask转换成.png文件
        # 保存掩码到 cache_dir
        file_name = os.path.basename(source_image_path).split(".")[0]
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # # 将掩码转换为图像并保存
        # sep_face_mask_img = Image.fromarray(sep_face_mask)
        # sep_face_mask_img.save(os.path.join(cache_dir, f"{file_name}_sep_face.png"))

        # sep_body_mask_img = Image.fromarray(sep_body_mask)
        # sep_body_mask_img.save(os.path.join(cache_dir, f"{file_name}_sep_body.png"))

        sep_pose_mask_img = Image.fromarray(sep_pose_mask)
        sep_pose_mask_img.save(os.path.join(cache_dir, f"{file_name}_sep_pose.png"))

        # body_mask_img = Image.fromarray(body_mask)
        # body_mask_img.save(os.path.join(cache_dir, f"{file_name}_body.png"))

        # 从 cache_dir 中读取掩码
        # sep_face_mask = Image.open(os.path.join(cache_dir, f"{file_name}_sep_face.png"))
        # sep_body_mask = Image.open(os.path.join(cache_dir, f"{file_name}_sep_body.png"))
        sep_pose_mask = Image.open(os.path.join(cache_dir, f"{file_name}_sep_pose.png"))
        # body_mask_pil = Image.open(os.path.join(cache_dir, f"{file_name}_body.png")).convert("RGB")
        # body_mask = self._augmentation(body_mask_pil, self.cond_transform)
        ################################################################



        # pixel_values_face_mask = [
        #     self._augmentation(sep_face_mask, self.attn_transform_64),
        #     self._augmentation(sep_face_mask, self.attn_transform_32),
        #     self._augmentation(sep_face_mask, self.attn_transform_16),
        #     self._augmentation(sep_face_mask, self.attn_transform_8),
        # ]
        # pixel_values_bodys_mask = [
        #     self._augmentation(sep_body_mask, self.attn_transform_64),
        #     self._augmentation(sep_body_mask, self.attn_transform_32),
        #     self._augmentation(sep_body_mask, self.attn_transform_16),
        #     self._augmentation(sep_body_mask, self.attn_transform_8),
        # ]
        pixel_values_full_mask = [
            self._augmentation(sep_pose_mask, self.attn_transform_64),
            self._augmentation(sep_pose_mask, self.attn_transform_32),
            self._augmentation(sep_pose_mask, self.attn_transform_16),
            self._augmentation(sep_pose_mask, self.attn_transform_8),
        ]

        pixel_values_full_mask = [mask.view(1, -1)
                                  for mask in pixel_values_full_mask]
        # pixel_values_face_mask = [mask.view(1, -1)
        #                           for mask in pixel_values_face_mask]
        # pixel_values_bodys_mask = [mask.view(1, -1)
        #                          for mask in pixel_values_bodys_mask]

        return pixel_values_ref_img, clip_image_embeds, pixel_values_full_mask


    def preprocess_1(self, source_image_path: str, cache_dir: str, face_region_ratio: float):
        """
        Apply preprocessing to the source image to prepare for face analysis.

        Parameters:
            source_image_path (str): The path to the source image.
            cache_dir (str): The directory to cache intermediate results.

        Returns:
            None
        """
        image_encoder_path = './pretrained_models/sd-image-variations-diffusers/image_encoder'
        image_enc = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path,
        ).to(dtype=torch.float16, device="cuda")

        ref_image_pil = Image.open(source_image_path).convert("RGB")
        pixel_values_ref_img = self._augmentation(ref_image_pil, self.pixel_transform)
        # 1. image augmentation
        clip_image = self.clip_image_processor(
            images=ref_image_pil, return_tensors="pt"
        ).pixel_values[0] 

        clip_image = clip_image.unsqueeze(0)
        clip_image_embeds = image_enc(
            clip_image.to("cuda", dtype=torch.float16)
        ).image_embeds

        # 使用yolov5生成mask
        ################################################################
        face_mask, body_mask = process_single_images(source_image_path)
        # 4 gaussian blur
        blur_face_mask = blur_mask(face_mask, (64, 64), (41, 41))
        blur_body_mask = blur_mask(body_mask, (64, 64), (71, 71))

        # 5 seperate mask 
        sep_face_mask = blur_face_mask
        sep_body_mask = cv2.subtract(blur_body_mask, sep_face_mask)
        sep_pose_mask = 255.0 - blur_face_mask

        sep_face_mask = sep_face_mask.astype(np.uint8)
        sep_body_mask = sep_body_mask.astype(np.uint8)
        sep_pose_mask = sep_pose_mask.astype(np.uint8)
        # 将mask转换成.png文件
        # 保存掩码到 cache_dir
        file_name = os.path.basename(source_image_path).split(".")[0]

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # 将掩码转换为图像并保存
        sep_face_mask_img = Image.fromarray(sep_face_mask)
        sep_face_mask_img.save(os.path.join(cache_dir, f"{file_name}_sep_face.png"))

        sep_body_mask_img = Image.fromarray(sep_body_mask)
        sep_body_mask_img.save(os.path.join(cache_dir, f"{file_name}_sep_body.png"))

        sep_pose_mask_img = Image.fromarray(sep_pose_mask)
        sep_pose_mask_img.save(os.path.join(cache_dir, f"{file_name}_sep_pose.png"))

        body_mask_img = Image.fromarray(body_mask)
        body_mask_img.save(os.path.join(cache_dir, f"{file_name}_body.png"))

        # 从 cache_dir 中读取掩码
        sep_face_mask = Image.open(os.path.join(cache_dir, f"{file_name}_sep_face.png"))
        sep_body_mask = Image.open(os.path.join(cache_dir, f"{file_name}_sep_body.png"))
        sep_pose_mask = Image.open(os.path.join(cache_dir, f"{file_name}_sep_pose.png"))
        body_mask_pil = Image.open(os.path.join(cache_dir, f"{file_name}_body.png")).convert("RGB")
        body_mask = self._augmentation(body_mask_pil, self.cond_transform)
        ################################################################



        pixel_values_face_mask = [
            self._augmentation(sep_face_mask, self.attn_transform_64),
            self._augmentation(sep_face_mask, self.attn_transform_32),
            self._augmentation(sep_face_mask, self.attn_transform_16),
            self._augmentation(sep_face_mask, self.attn_transform_8),
        ]
        pixel_values_bodys_mask = [
            self._augmentation(sep_body_mask, self.attn_transform_64),
            self._augmentation(sep_body_mask, self.attn_transform_32),
            self._augmentation(sep_body_mask, self.attn_transform_16),
            self._augmentation(sep_body_mask, self.attn_transform_8),
        ]
        pixel_values_full_mask = [
            self._augmentation(sep_pose_mask, self.attn_transform_64),
            self._augmentation(sep_pose_mask, self.attn_transform_32),
            self._augmentation(sep_pose_mask, self.attn_transform_16),
            self._augmentation(sep_pose_mask, self.attn_transform_8),
        ]

        pixel_values_full_mask = [mask.view(1, -1)
                                  for mask in pixel_values_full_mask]
        pixel_values_face_mask = [mask.view(1, -1)
                                  for mask in pixel_values_face_mask]
        pixel_values_bodys_mask = [mask.view(1, -1)
                                 for mask in pixel_values_bodys_mask]

        return pixel_values_ref_img, clip_image_embeds, pixel_values_full_mask, pixel_values_face_mask, pixel_values_bodys_mask

    def preprocess_mov_mask(self, face_masks_list: list, lips_masks_list: list, face_region_ratio: float, clip_length):
 
        assert face_masks_list[0] is not None, "Fail to load face mask."
        assert lips_masks_list[0] is not None, "Fail to load lip mask."

        pixel_values_face_mask = [
            self._augmentation(face_masks_list, self.attn_transform_64),
            self._augmentation(face_masks_list, self.attn_transform_32),
            self._augmentation(face_masks_list, self.attn_transform_16),
            self._augmentation(face_masks_list, self.attn_transform_8),
        ]
        pixel_values_lips_mask = [
            self._augmentation(lips_masks_list, self.attn_transform_64),
            self._augmentation(lips_masks_list, self.attn_transform_32),
            self._augmentation(lips_masks_list, self.attn_transform_16),
            self._augmentation(lips_masks_list, self.attn_transform_8),
        ]

        pixel_values_face_mask = [mask.view(clip_length, 1, -1).squeeze(1)
                                  for mask in pixel_values_face_mask]
        pixel_values_lips_mask = [mask.view(clip_length, 1, -1).squeeze(1)
                                 for mask in pixel_values_lips_mask]    
        return pixel_values_face_mask, pixel_values_lips_mask






    def close(self):
        """
        Closes the ImageProcessor and releases any resources held by the FaceAnalysis instance.

        Args:
            self: The ImageProcessor instance.

        Returns:
            None.
        """
        for _, model in self.face_analysis.models.items():
            if hasattr(model, "Dispose"):
                model.Dispose()

    def _augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()


class ImageProcessorForDataProcessing():
    """
    ImageProcessor is a class responsible for processing images, particularly for face-related tasks.
    It takes in an image and performs various operations such as augmentation, face detection,
    face embedding extraction, and rendering a face mask. The processed images are then used for
    further analysis or recognition purposes.

    Attributes:
        img_size (int): The size of the image to be processed.
        face_analysis_model_path (str): The path to the face analysis model.

    Methods:
        preprocess(source_image_path, cache_dir):
            Preprocesses the input image by performing augmentation, face detection,
            face embedding extraction, and rendering a face mask.

        close():
            Closes the ImageProcessor and releases any resources being used.

        _augmentation(images, transform, state=None):
            Applies image augmentation to the input images using the given transform and state.

        __enter__():
            Enters a runtime context and returns the ImageProcessor object.

        __exit__(_exc_type, _exc_val, _exc_tb):
            Exits a runtime context and bodyles any exceptions that occurred during the processing.
    """
    def __init__(self, face_analysis_model_path, landmark_model_path, step) -> None:
        # if step == 2:
        #     # self.face_analysis = FaceAnalysis(
        #     #     name="",
        #     #     root=face_analysis_model_path,
        #     #     providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        #     # )
        #     # self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))
        # #     self.image_enc = CLIPVisionModelWithProjection.from_pretrained(
        # # "pretrained_models/sd-image-variations-diffusers/image_encoder",).to(dtype=torch.float16, device="cuda")
        # #     self.landmarker = None
        # else:
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        # Create a face landmarker instance with the video mode:
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=landmark_model_path),
            running_mode=VisionRunningMode.IMAGE,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        self.face_analysis = None

        self.mode_path = 'pretrained_models/yolov5s.pt'

    


    def preprocess(self, source_image_path: str):
        """
        Apply preprocessing to the source image to prepare for face analysis.

        Parameters:
            source_image_path (str): The path to the source image.
            cache_dir (str): The directory to cache intermediate results.

        Returns:
            None
        """
        # 1. get pose embdeding
        face_mask, clip_image_embeds, sep_pose_mask, sep_face_mask, sep_body_mask = None, None, None, None, None
        
        if self.face_analysis:
            for frame in sorted(os.listdir(source_image_path)):
                try:
                    source_image = Image.open(
                        os.path.join(source_image_path, frame))
                    ref_image_pil = source_image.convert("RGB")
                    # 2.1 detect face
                    faces = self.face_analysis.get(cv2.cvtColor(
                        np.array(ref_image_pil.copy()), cv2.COLOR_RGB2BGR))
                    # use max size face
                    face = sorted(faces, key=lambda x: (
                        x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1]
                    # 2.2 face embedding
                    face_emb = face["embedding"]
                    if face_emb is not None:
                        break
                except Exception as _:
                    continue

        

        if self.landmarker:
            # 3.1 get face landmark
            # landmarks, height, width = get_landmark_overframes(
            #     self.landmarker, source_image_path)


            # body_landmarks = get_body_landmark_overframes(source_image_path) 

            # if len(landmarks) == 0:
            #     print("未检测到人脸，用body_landmark代替")
            #     landmarks = body_landmarks
            # # body_landmarks, _, _ = get_body_landmark_overframes(
            # #     self.landmarker_body, source_image_path)
            

            
            # assert len(landmarks) == len(os.listdir(source_image_path))
            # assert len(body_landmarks) == len(os.listdir(source_image_path))

            # # 3 render face and lip mask
            # face_mask = get_union_face_mask(landmarks, height, width)
            # # body_mask = get_union_body_mask(landmarks, height, width)
            # body_mask = get_union_body_mask(body_landmarks, height, width)
            # # 3.1 get body landmark


            face_mask, body_mask, lip_mask = process_images_in_folder(source_image_path)


            # 4 gaussian blur
            blur_face_mask = blur_mask(face_mask, (64, 64), (41, 41))
            blur_body_mask = blur_mask(body_mask, (64, 64), (61, 61))
            blur_lip_mask = blur_mask(lip_mask, (64, 64), (31, 31))

            # 5 seperate mask 
            sep_face_mask = blur_face_mask
            sep_body_mask = cv2.subtract(blur_body_mask, sep_face_mask)
            sep_pose_mask = 255.0 - blur_face_mask
            sep_lip_mask = blur_lip_mask

            # sep_lip_mask = blur_lip_mask

        return body_mask, clip_image_embeds, sep_lip_mask, sep_face_mask, sep_body_mask



    # def close(self):
    #     """
    #     Closes the ImageProcessor and releases any resources held by the FaceAnalysis instance.

    #     Args:
    #         self: The ImageProcessor instance.

    #     Returns:
    #         None.
    #     """
    #     for _, model in self.face_analysis.models.items():
    #         if hasattr(model, "Dispose"):
    #             model.Dispose()

    def _augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()


