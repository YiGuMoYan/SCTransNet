import random
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torchvision.transforms import ToPILImage

import model.Config as config
from dataset import TestSetLoader
from model.SCTransNet import SCTransNet

state_dict = torch.load("log/SIRST3/SCTransNet_537_best.pth.tar")
config_vit = config.get_SCTrans_config()
new_state_dict = OrderedDict()
for k, v in state_dict['state_dict'].items():
    name = k[6:]
    new_state_dict[name] = v
model = SCTransNet(config_vit)
model.load_state_dict(new_state_dict)

dataset_name = "SIRST3"

test_dataset_loader = TestSetLoader('datasets/', dataset_name, dataset_name)

img, gt_mask, size, img_dir = random.choice(test_dataset_loader)


def is_overlapping(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # 检查是否在x轴上重叠
    if x1 + w1 <= x2 or x2 + w2 <= x1:
        return False

    # 检查是否在y轴上重叠
    if y1 + h1 <= y2 or y2 + h2 <= y1:
        return False

    return True


def infer(raw_img, raw_mask, lower=10):
    # tensor to pil
    to_pil = ToPILImage()

    # 获取raw的pil cv
    raw_pil = to_pil(raw_img)
    raw_pil.save("raw.jpg")
    raw_pil_array = np.array(raw_pil)
    raw_cv = cv2.cvtColor(raw_pil_array, cv2.COLOR_RGB2BGR)

    # 输出原生遮罩
    raw_mask_pil = to_pil(raw_mask)
    raw_mask_pil.save("raw_mask.jpg")
    raw_mask_array = np.array(raw_mask_pil)
    raw_mask_cv = cv2.cvtColor(raw_mask_array, cv2.COLOR_RGB2BGR)

    model.eval()

    # 创建结果numpy
    res_np = np.zeros(raw_img.shape)

    with torch.no_grad():
        # 获取结果
        res = model(raw_img.unsqueeze(0))
        # 取每个遮罩的最大值
        for i in range(len(res)):
            infer = res[i][0]
            res_np = np.maximum(res_np, infer.numpy())

        res_np = (res_np * 255).astype(np.uint8)
        res_cv = res_np[0]

        # 二值化
        res_cv = cv2.inRange(res_cv, lower, 255)
        res_mask = cv2.cvtColor(res_cv, cv2.COLOR_GRAY2BGR)

        # 寻找轮廓
        contours, _ = cv2.findContours(res_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        target_contours, _ = cv2.findContours(cv2.cvtColor(raw_mask_cv, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
        target_contour_areas = [(cv2.boundingRect(target_contour)) for target_contour in target_contours]

        res_cv = raw_cv.copy()
        for contour in contours:
            # 计算轮廓的边界矩形
            x, y, w, h = cv2.boundingRect(contour)
            # 绘制矩形框
            color = (0, 0, 255)
            for target_contour_area in target_contour_areas:
                if is_overlapping((x, y, w, h), target_contour_area):
                    color = (0, 255, 0)
            thickness = 2  # 线条粗细
            cv2.rectangle(res_cv, (x, y), (x + w, y + h), color, thickness)
            cv2.rectangle(res_mask, (x, y), (x + w, y + h), color, thickness)

        cv2.imwrite('res.jpg', res_cv)
        cv2.imwrite('res_mask.jpg', res_mask)

        width, height, _ = res_cv.shape
        gap = 10
        result_height = 2 * height + gap
        result_width = 2 * width + gap
        result = np.zeros((result_height, result_width, 3), dtype=np.uint8)
        result[0:height, 0:width] = raw_cv
        result[0:height, width + gap:2 * width + gap] = raw_mask_cv
        result[height + gap:2 * height + gap, 0:width] = res_cv
        result[height + gap:2 * height + gap, width + gap:2 * width + gap] = res_mask
        cv2.imwrite("result.jpg", result)


infer(img, gt_mask, lower=7)
