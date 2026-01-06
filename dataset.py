import json
import numpy as np
import cv2
from typing import List, Dict, Tuple
import os

def parse_labelme_json(json_path: str, image_shape: Tuple[int, int] = None):
    """
    解析LabelMe JSON文件，提取多边形标注
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取图像尺寸
    if image_shape is None:
        height = data['imageHeight']
        width = data['imageWidth']
    else:
        height, width = image_shape
    
    # 初始化掩码字典
    masks_dict = {}
    categories = []
    
    # 处理每个标注
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        shape_type = shape.get('shape_type', 'polygon')
        
        # 创建空掩码
        if label not in masks_dict:
            masks_dict[label] = np.zeros((height, width), dtype=np.uint8)
            categories.append(label)
        
        # 转换点坐标为整数
        pts = np.array(points, dtype=np.int32)
        
        if shape_type == 'polygon':
            # 填充多边形
            cv2.fillPoly(masks_dict[label], [pts], 1)
        elif shape_type == 'rectangle':
            # 矩形
            x1, y1 = np.min(pts, axis=0)
            x2, y2 = np.max(pts, axis=0)
            masks_dict[label][y1:y2, x1:x2] = 1
        elif shape_type == 'circle':
            # 圆形（近似为椭圆）
            center = np.mean(pts, axis=0).astype(int)
            radius = int(np.linalg.norm(pts[0] - pts[1]))
            cv2.circle(masks_dict[label], tuple(center), radius, 1, -1)
    
    return masks_dict, categories

def draw_gt_mask_from_json(json_path):
    masks_dict, categories = parse_labelme_json(json_path)
    if(len(categories)>1):
        mask = masks_dict['1']-masks_dict['0']
    else:
        mask = masks_dict['1']
    mask *= 255
    cv2.imwrite('output/gt_mask.png', mask)

def cal_miou(pre_mask_folder, gt_mask_json_folder, order):


    pre_mask_path = os.path.join(pre_mask_folder, order) + '.png'
    gt_mask_json = os.path.join(gt_mask_json_folder, order) + '.json'

    masks_dict, categories = parse_labelme_json(gt_mask_json)
    if(len(categories)>1):
        gt_mask = masks_dict['1']-masks_dict['0']
    else:
        gt_mask = masks_dict['1']

    pre_mask = cv2.imread(pre_mask_path, cv2.IMREAD_GRAYSCALE)
    intersection = np.logical_and(pre_mask, gt_mask).sum()
    union = np.logical_or(pre_mask, gt_mask).sum()
    return intersection/union

pre_mask_folder = r'D:\zhj\code\datapath\masks'
gt_mask_json_folder = r'D:\zhj\code\datapath\regions_json'
print(cal_miou(pre_mask_folder, gt_mask_json_folder, r'59'))