import torch
import cv2
import numpy as np
import os

def load_yolov5_model(weights_path: str):
    """
    Load YOLOv5 model from the specified weights path.
    """
    # 加载模型架构和权重
    model = torch.hub.load('yolov5', 'custom', path=weights_path, source='local')
    model.eval()
    return model

def detect_and_generate_masks(image_path, model):
    """
    使用YOLOv5检测图像中人的位置并生成人脸和人体的mask。

    Args:
        image_path (str): 输入图像的路径。
        model: 预加载的 YOLOv5 模型。

    Returns:
        tuple: 人脸和人体的mask。
    """
    model.classes = [0]  # 仅检测人类（YOLOv5类别0是人类）
    if isinstance(image_path, os.PathLike):
        image_path = str(image_path)
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to open image: {image_path}.")
        return None, None

    # 转换图像颜色空间（OpenCV使用BGR，但YOLOv5需要RGB）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 使用模型进行预测
    results = model(image_rgb)

    # 获取检测结果
    detections = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, confidence, class

    # 创建一个全黑的mask
    human_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    face_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    lips_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    max_human_area = 0
    max_face_area = 0

    # 遍历检测结果，绘制矩形框
    for detection in detections:
        x1, y1, x2, y2, confidence, cls = detection
        if confidence > 0.5:  # 仅保留置信度大于0.5的检测结果
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            human_area = (x2 - x1) * (y2 - y1)
            if human_area > max_human_area:
                max_human_area = human_area
                human_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                human_mask[y1:y2, x1:x2] = 255

                # 简单假设人脸位于人体矩形框的上半部分，生成一个人脸mask

                # TODO 人脸矩形框的宽度为人体矩形框的0.5倍且位置居中
                
                # TODO 设置嘴唇的mask 为人脸矩形框的下半部分，宽度为人脸矩形框的0.5倍且居中

                # 人脸矩形框的宽度为人体矩形框的0.5倍且位置居中
                face_width = int((x2 - x1) * 0.8)
                face_height = int((y2 - y1) / 2)
                face_x1 = x1 + (x2 - x1 - face_width) // 2
                face_x2 = face_x1 + face_width
                face_y2 = y1 + face_height

                face_area = face_width * face_height
                if face_area > max_face_area:
                    max_face_area = face_area
                    face_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    face_mask[y1:face_y2, face_x1:face_x2] = 255

                    # 设置嘴唇的mask为人脸矩形框的下半部分，宽度为人脸矩形框的0.5倍且居中
                    lips_height = face_height // 2
                    lips_width = face_width // 2
                    lips_x1 = face_x1 + (face_width - lips_width) // 2
                    lips_x2 = lips_x1 + lips_width
                    lips_y1 = y1 + face_height // 2
                    lips_y2 = lips_y1 + lips_height

                    lips_mask[lips_y1:lips_y2, lips_x1:lips_x2] = 255

    return face_mask, human_mask, lips_mask

def process_images_in_folder(folder_path):
    """
    处理文件夹中的前五张图片，选出最大的脸部和人体的mask作为输出。

    Args:
        folder_path (str): 包含图片的文件夹路径。

    Returns:
        tuple: 最大的脸部mask和人体mask。
    """
    image_files = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_files) < 5:
        raise ValueError("文件夹中图片数量不足5张。")

    max_face_mask = None
    max_body_mask = None
    max_lip_mask = None
    max_face_area = 0
    max_body_area = 0
    max_lip_area = 0
    model_path = '/root/node03-nfs/aaai/hallo/pretrained_models/yolov5s.pt'
    model = load_yolov5_model(model_path)

    for image_file in image_files[:20]:
        face_mask, body_mask, lip_mask = detect_and_generate_masks(image_file, model)
        if face_mask is not None and body_mask is not None:
            face_area = np.sum(face_mask == 255)
            body_area = np.sum(body_mask == 255)
            lip_area = np.sum(lip_mask == 255)
            if face_area > max_face_area:
                max_face_area = face_area
                max_face_mask = face_mask
            if body_area > max_body_area:
                max_body_area = body_area
                max_body_mask = body_mask
            if lip_area > max_lip_area:
                max_lip_area = lip_area
                max_lip_mask = lip_mask

    return max_face_mask, max_body_mask, max_lip_mask



def process_single_images(image_path):
    
    """
    处理推理图片，选出脸部和人体的mask作为输出。

    Args:
        folder_path (str): 包含图片的文件夹路径。

    Returns:
        tuple: 最大的脸部mask和人体mask。
    """

    model_path = '/root/node03-nfs/aaai/hallo/pretrained_models/yolov5s.pt'
    model = load_yolov5_model(model_path)
    face_mask, body_mask, _ = detect_and_generate_masks(image_path, model)
    return face_mask, body_mask