# https://github.com/IDEA-Research/DWPose
# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import copy
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import torch
from controlnet_aux.util import HWC3, resize_image
from PIL import Image

from . import util
from .wholebody import Wholebody


def draw_pose(pose, H, W):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas = util.draw_facepose(canvas, faces)

    return canvas


class DWposeDetector:
    def __init__(self):
        pass

    def to(self, device):
        self.pose_estimation = Wholebody(device)
        return self

    def cal_height(self, input_image):
        input_image = cv2.cvtColor(
            np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR
        )

        input_image = HWC3(input_image)
        H, W, C = input_image.shape
        with torch.no_grad():
            candidate, subset , _ = self.pose_estimation(input_image)
            nums, keys, locs = candidate.shape
            # candidate[..., 0] /= float(W)
            # candidate[..., 1] /= float(H)
            body = candidate
        return body[0, ..., 1].min(), body[..., 1].max() - body[..., 1].min()

    def __call__(
        self,
        input_image,
        detect_resolution=512,
        image_resolution=512,
        output_type="pil",
        **kwargs,
    ):
        input_image = cv2.cvtColor(
            np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR
        )

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        H, W, C = input_image.shape
        with torch.no_grad():
            candidate, subset, whole_keypoints = self.pose_estimation(input_image)

            whole_keypoints = whole_keypoints.copy()
            
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            score = subset[:, :18]
            max_ind = np.mean(score, axis=-1).argmax(axis=0)
            score = score[[max_ind]]
            body = candidate[:, :18].copy()
            body = body[[max_ind]]
            nums = 1
            body = body.reshape(nums * 18, locs)
            body_score = copy.deepcopy(score)
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1

            un_visible = subset < 0.3
            candidate[un_visible] = -1

            foot = candidate[:, 18:24]

            faces = candidate[[max_ind], 24:92]
            # mask = np.ones(faces.shape, dtype=np.uint8)
            # lip_indices = range(48, 68)
            # mask[lip_indices, :] = 0    
            # faces = faces * mask 

            hands = candidate[[max_ind], 92:113]
            hands = np.vstack([hands, candidate[[max_ind], 113:]])

            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)


            detected_map = draw_pose(pose, H, W)
            detected_map = HWC3(detected_map)

            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape

            detected_map = cv2.resize(
                detected_map, (W, H), interpolation=cv2.INTER_LINEAR
            )

            if output_type == "pil":
                detected_map = Image.fromarray(detected_map)

            return detected_map, body_score, whole_keypoints

def draw_pose(pose, H, W):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]

    # 将背景设置为纯白
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)
    canvas = util.draw_handpose(canvas, hands)
    canvas = util.draw_facepose(canvas, faces)

    return canvas

def draw_pose_mask_head(pose, H, W):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    # canvas = util.draw_bodypose(canvas, candidate, subset)

    # canvas = util.draw_handpose_with_individual_bbox(canvas, hands)

    canvas = util.draw_facepose_with_bbox(canvas, faces)

    return canvas


def draw_pose_mask_lips(pose, H, W):
    bodies = pose["bodies"]
    faces = pose["faces"]
    lips = pose["lips"]
    hands = pose["hands"]
    eyes = pose["eyes"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    
    # canvas = util.draw_bodypose(canvas, candidate, subset)

    # canvas = util.draw_handpose_with_individual_bbox(canvas, hands)
    # canvas = util.draw_handpose_with_individual_bbox(canvas, eyes)


    canvas = util.draw_facepose_with_bbox(canvas, lips)

    return canvas

def draw_pose_mask_hand(pose, H, W):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    # canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose_with_individual_bbox(canvas, hands)

    # canvas = util.draw_facepose_with_bbox(canvas, faces)

    return canvas



class DWposeDetector_movment_mask:
    def __init__(self):
        pass


    def to(self, device):
        self.pose_estimation = Wholebody(device)
        return self
    
    def cal_height(self, input_image):
        input_image = cv2.cvtColor(
            np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR
        )   

    def __call__(
        self,
        candidate, # keypoints
        subset,    # scores
        output_type="pil",
        **kwargs,
    ):
            
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(512)
            candidate[..., 1] /= float(512)
            score = subset[:, :18]
            max_ind = np.mean(score, axis=-1).argmax(axis=0)
            score = score[[max_ind]]
            body = candidate[:, :18].copy()
            body = body[[max_ind]]
            nums = 1
            body = body.reshape(nums * 18, locs)
            body_score = copy.deepcopy(score)
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1

            un_visible = subset < 0.3
            candidate[un_visible] = -1
            # mask = np.ones(candidate.shape, dtype=np.uint8)  


            foot = candidate[:, 18:24]

            faces = candidate[[max_ind], 24:92]
            eyes = candidate[[max_ind], 60:72]
            right_eye_indices = candidate[[max_ind], 66:72]

            hands = candidate[[max_ind], 92:113]

            lips = candidate[[max_ind], 72:92]

            hands = np.vstack([hands, candidate[[max_ind], 113:]])

            bodies = dict(candidate=body, subset=score)

            lip_indices = range(48, 68)
            # mask[:, lip_indices, :] = 0
            # faces_without_lips = faces * mask 


            pose = dict(bodies=bodies, hands=hands, faces=faces, lips=lips, eyes=eyes)
            # pose_1 = dict(bodies=bodies, hands=hands, faces=faces_without_lips, eyes=eyes)
            detected_map_lips_eyes = draw_pose_mask_lips(pose, 512, 512)
            # detected_map_lips = draw_pose_mask_lips(pose, 512, 512)
            # detected_map_lips = draw_pose_mask_lips(pose, 512, 512)
            detected_map_face = draw_pose_mask_head(pose, 512, 512)
            detected_map_pose = draw_pose(pose, 512, 512)

            detected_map_hands = draw_pose_mask_hand(pose, 512, 512)
            # detected_map_face_hand = detected_map_face_hand - detected_map_lips_eyes
            detected_map_face = detected_map_face + detected_map_hands
            # detected_map_lips = process_pose(detected_map_lips, output_type)
            detected_map_pose = process_pose(detected_map_pose, output_type)
            detected_map_face = process_pose(detected_map_face, output_type)
            detected_map_hands = process_pose(detected_map_hands, output_type)
            detected_map_lips_eyes = process_pose(detected_map_lips_eyes, output_type)
            # detected_map_faces = process_pose(detected_map_faces, output_type)
            return detected_map_pose, detected_map_hands, detected_map_lips_eyes, detected_map_face, body_score
    
def process_pose(detected_map, output_type):
    detected_map = HWC3(detected_map)
    detected_map = cv2.resize(
        detected_map, (512, 512), interpolation=cv2.INTER_LINEAR
    )
    if output_type == "pil":
        detected_map = Image.fromarray(detected_map)
    return detected_map