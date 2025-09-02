# https://github.com/IDEA-Research/DWPose
import math
import numpy as np
import matplotlib
import cv2
import matplotlib.colors as mcolors

eps = 0.01


def smart_resize(x, s):
    Ht, Wt = s
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(
            x,
            (int(Wt), int(Ht)),
            interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4,
        )
    else:
        return np.stack([smart_resize(x[:, :, i], s) for i in range(Co)], axis=2)


def smart_resize_k(x, fx, fy):
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    Ht, Wt = Ho * fy, Wo * fx
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(
            x,
            (int(Wt), int(Ht)),
            interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4,
        )
    else:
        return np.stack([smart_resize_k(x[:, :, i], fx, fy) for i in range(Co)], axis=2)


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights[
            ".".join(weights_name.split(".")[1:])
        ]
    return transfered_model_weights


def draw_bodypose(canvas, candidate, subset):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18],
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.9).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas


def draw_handpose(canvas, all_hand_peaks, eps=1e-5):
    """
    all_hand_peaks: List[np.ndarray | list]，
        每只手 21 个关键点，坐标已归一化到 [0,1]。
    canvas: H×W×3, uint8
    """
    H, W, _ = canvas.shape

    # --- 21 点骨架连线（Mediapipe HandPose 兼容） -------------------
    edges = [
        (0, 1),  (1, 2),  (2, 3),  (3, 4),      # 拇指
        (0, 5),  (5, 6),  (6, 7),  (7, 8),      # 食指
        (0, 9),  (9,10), (10,11), (11,12),      # 中指
        (0,13), (13,14), (14,15), (15,16),      # 无名指
        (0,17), (17,18), (18,19), (19,20),      # 小指
    ]

    # --- 为每条边准备独立颜色（HSV 均分 → BGR‑uint8） ---------------
    n_edges = len(edges)
    edge_colors = []
    for i in range(n_edges):
        rgb = mcolors.hsv_to_rgb([i / n_edges, 1.0, 1.0]) * 255
        bgr = rgb[::-1].astype(np.uint8)        # RGB → BGR，float → uint8
        edge_colors.append(tuple(int(c) for c in bgr))

    # --- 逐只手绘制 -----------------------------------------------
    for peaks in all_hand_peaks:
        peaks = np.asarray(peaks, dtype=np.float32)

        # 1. 画骨架线
        for idx, (p1, p2) in enumerate(edges):
            x1, y1 = peaks[p1]
            x2, y2 = peaks[p2]
            x1, y1 = int(x1 * W), int(y1 * H)
            x2, y2 = int(x2 * W), int(y2 * H)

            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(canvas, (x1, y1), (x2, y2),
                         edge_colors[idx], thickness=2)

        # 2. 画关键点
        for x_norm, y_norm in peaks:
            x, y = int(x_norm * W), int(y_norm * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)

    return canvas

def draw_handpose_with_individual_bbox(canvas, all_hand_peaks):
    """
    绘制手部关键点，并为每只手单独绘制边界框并将范围置为1
    """
    H, W, C = canvas.shape
    for peaks in all_hand_peaks:
        peaks = np.array(peaks)
        min_x, min_y = W, H
        max_x, max_y = 0, 0

        # 计算单只手的边界框
        for keypoint in peaks:
            x, y = keypoint
            x = int(x * W)
            y = int(y * H)
            if x > 0 and y > 0:  # 忽略无效点
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

        # 如果边界框有效，将范围置为1
        if min_x < max_x and min_y < max_y:
            canvas[min_y:max_y, min_x:max_x, :] = 255
    return canvas


# def draw_handpose_with_individual_bbox(canvas, all_hand_peaks, min_bbox_size=80):
#     """
#     绘制手部关键点，并为每只手单独绘制边界框，将范围置为1，
#     同时确保边界框是正方形且大于等于给定的最小尺寸。
    
#     :param canvas: 画布（H, W, C）图像
#     :param all_hand_peaks: 所有手部关键点
#     :param min_bbox_size: 最小边界框的尺寸（正方形），单位为像素
#     :return: 更新后的canvas
#     """
#     H, W, C = canvas.shape
#     for peaks in all_hand_peaks:
#         peaks = np.array(peaks)
#         min_x, min_y = W, H
#         max_x, max_y = 0, 0

#         # 计算单只手的边界框
#         for keypoint in peaks:
#             x, y = keypoint
#             x = int(x * W)  # 转换为实际像素坐标
#             y = int(y * H)
#             if x > 0 and y > 0:  # 忽略无效点
#                 min_x = min(min_x, x)
#                 min_y = min(min_y, y)
#                 max_x = max(max_x, x)
#                 max_y = max(max_y, y)

#         # 如果边界框有效
#         if min_x < max_x and min_y < max_y:
#             # 计算当前边界框的宽度和高度
#             bbox_width = max_x - min_x
#             bbox_height = max_y - min_y

#             # 确保边界框是正方形且大于最小尺寸
#             max_side = max(bbox_width, bbox_height)
#             if max_side < min_bbox_size:
#                 # 如果边界框的长宽都小于最小值，则调整为最小尺寸
#                 delta = min_bbox_size - max_side
#                 if bbox_width < bbox_height:
#                     max_x += delta  # 增加宽度
#                 else:
#                     max_y += delta  # 增加高度

#             # 如果需要保证边界框为正方形，调整更大的边
#             if bbox_width < bbox_height:
#                 max_x = min_x + (max_y - min_y)
#             else:
#                 max_y = min_y + (max_x - min_x)

#             # 将计算得到的边界框区域设置为255
#             canvas[min_y:max_y, min_x:max_x, :] = 255
#     return canvas




def draw_facepose(canvas, all_lmks):
    H, W, C = canvas.shape
    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas


# def draw_facepose(canvas, all_lmks):
#     """
#     all_lmks: List[np.ndarray | list]，每张脸的关键点坐标，归一化到 [0,1]
#     """
#     H, W, _ = canvas.shape
#     eps = 1e-5

#     # --- 颜色（BGR） --------------------------------------------------
#     COLOR_EDGE  = (  0,   0, 0)   # 红：jaw line / 边缘
#     COLOR_NOSE  = (  0,   0, 0)   # 蓝：鼻子
#     COLOR_EYES  = (  0, 0,   0)   # 绿：双眼
#     COLOR_MOUTH = (  0, 0, 0)   # 黄：嘴巴

#     # --- 关键点分区 ---------------------------------------------------
#     idx_edge  = set(range(0, 17))          # 0‑16
#     idx_nose  = set(range(27, 36))         # 27‑35
#     idx_eyes  = set(range(36, 48))         # 36‑47
#     idx_mouth = set(range(48, 68))         # 48‑67

#     for lmks in all_lmks:
#         lmks = np.asarray(lmks, dtype=np.float32)

#         for i, (x_norm, y_norm) in enumerate(lmks):
#             x = int(x_norm * W)
#             y = int(y_norm * H)
#             if x <= eps or y <= eps:
#                 continue

#             # 选颜色
#             if i in idx_edge:
#                 color = COLOR_EDGE
#             elif i in idx_nose:
#                 color = COLOR_NOSE
#             elif i in idx_eyes:
#                 color = COLOR_EYES
#             elif i in idx_mouth:
#                 color = COLOR_MOUTH
#             else:
#                 color = (0, 0, 0)     # 其它点：蓝

#             cv2.circle(canvas, (x, y), 3, color, thickness=-1)

#     return canvas


def draw_facepose_with_bbox(canvas, all_lmks):
    """
    绘制脸部关键点并将脸部范围置为1
    """
    H, W, C = canvas.shape
    face_bbox = None

    for lmks in all_lmks:
        lmks = np.array(lmks)
        min_x, min_y = W, H
        max_x, max_y = 0, 0

        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > 0 and y > 0:  # 忽略无效点
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

        # 更新脸部边界框
        if min_x < max_x and min_y < max_y:
            if face_bbox is None:
                face_bbox = [min_x, min_y, max_x, max_y]
            else:
                face_bbox = [
                    min(face_bbox[0], min_x),
                    min(face_bbox[1], min_y),
                    max(face_bbox[2], max_x),
                    max(face_bbox[3], max_y),
                ]
    # 如果找到脸部边界框，则将范围置为1
    if face_bbox:
        min_x, min_y, max_x, max_y = face_bbox
        canvas[min_y:max_y, min_x:max_x, :] = 255

    return canvas

# detect hand according to body pose keypoints
# please refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/hand/handDetector.cpp
def handDetect(candidate, subset, oriImg):
    # right hand: wrist 4, elbow 3, shoulder 2
    # left hand: wrist 7, elbow 6, shoulder 5
    ratioWristElbow = 0.33
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]
    for person in subset.astype(int):
        # if any of three not detected
        has_left = np.sum(person[[5, 6, 7]] == -1) == 0
        has_right = np.sum(person[[2, 3, 4]] == -1) == 0
        if not (has_left or has_right):
            continue
        hands = []
        # left hand
        if has_left:
            left_shoulder_index, left_elbow_index, left_wrist_index = person[[5, 6, 7]]
            x1, y1 = candidate[left_shoulder_index][:2]
            x2, y2 = candidate[left_elbow_index][:2]
            x3, y3 = candidate[left_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, True])
        # right hand
        if has_right:
            right_shoulder_index, right_elbow_index, right_wrist_index = person[
                [2, 3, 4]
            ]
            x1, y1 = candidate[right_shoulder_index][:2]
            x2, y2 = candidate[right_elbow_index][:2]
            x3, y3 = candidate[right_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, False])

        for x1, y1, x2, y2, x3, y3, is_left in hands:
            # pos_hand = pos_wrist + ratio * (pos_wrist - pos_elbox) = (1 + ratio) * pos_wrist - ratio * pos_elbox
            # handRectangle.x = posePtr[wrist*3] + ratioWristElbow * (posePtr[wrist*3] - posePtr[elbow*3]);
            # handRectangle.y = posePtr[wrist*3+1] + ratioWristElbow * (posePtr[wrist*3+1] - posePtr[elbow*3+1]);
            # const auto distanceWristElbow = getDistance(poseKeypoints, person, wrist, elbow);
            # const auto distanceElbowShoulder = getDistance(poseKeypoints, person, elbow, shoulder);
            # handRectangle.width = 1.5f * fastMax(distanceWristElbow, 0.9f * distanceElbowShoulder);
            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            # x-y refers to the center --> offset to topLeft point
            # handRectangle.x -= handRectangle.width / 2.f;
            # handRectangle.y -= handRectangle.height / 2.f;
            x -= width / 2
            y -= width / 2  # width = height
            # overflow the image
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            width1 = width
            width2 = width
            if x + width > image_width:
                width1 = image_width - x
            if y + width > image_height:
                width2 = image_height - y
            width = min(width1, width2)
            # the max hand box value is 20 pixels
            if width >= 20:
                detect_result.append([int(x), int(y), int(width), is_left])

    """
    return value: [[x, y, w, True if left hand else False]].
    width=height since the network require squared input.
    x, y is the coordinate of top left 
    """
    return detect_result


# Written by Lvmin
def faceDetect(candidate, subset, oriImg):
    # left right eye ear 14 15 16 17
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]
    for person in subset.astype(int):
        has_head = person[0] > -1
        if not has_head:
            continue

        has_left_eye = person[14] > -1
        has_right_eye = person[15] > -1
        has_left_ear = person[16] > -1
        has_right_ear = person[17] > -1

        if not (has_left_eye or has_right_eye or has_left_ear or has_right_ear):
            continue

        head, left_eye, right_eye, left_ear, right_ear = person[[0, 14, 15, 16, 17]]

        width = 0.0
        x0, y0 = candidate[head][:2]

        if has_left_eye:
            x1, y1 = candidate[left_eye][:2]
            d = max(abs(x0 - x1), abs(y0 - y1))
            width = max(width, d * 3.0)

        if has_right_eye:
            x1, y1 = candidate[right_eye][:2]
            d = max(abs(x0 - x1), abs(y0 - y1))
            width = max(width, d * 3.0)

        if has_left_ear:
            x1, y1 = candidate[left_ear][:2]
            d = max(abs(x0 - x1), abs(y0 - y1))
            width = max(width, d * 1.5)

        if has_right_ear:
            x1, y1 = candidate[right_ear][:2]
            d = max(abs(x0 - x1), abs(y0 - y1))
            width = max(width, d * 1.5)

        x, y = x0, y0

        x -= width
        y -= width

        if x < 0:
            x = 0

        if y < 0:
            y = 0

        width1 = width * 2
        width2 = width * 2

        if x + width > image_width:
            width1 = image_width - x

        if y + width > image_height:
            width2 = image_height - y

        width = min(width1, width2)

        if width >= 20:
            detect_result.append([int(x), int(y), int(width)])

    return detect_result


# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j
