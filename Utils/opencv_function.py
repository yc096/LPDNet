# _*_ coding: utf-8 _*_
# @Time : 2022/4/5 9:22 
# @Author : yc096
# @File : opencv_function.py
import cv2
import time
import torch
import numpy as np


def ndarrayImageToTensor(img):
    '''
    [H,W,[C]]的ndaray图像转为torch.tensor
    '''
    if img.ndim == 2:
        img = img[:, :, None]
    img = img.transpose((2, 0, 1))  # [H,W,C]->[C,H,W]
    img = torch.from_numpy(img).contiguous()
    if isinstance(img, torch.ByteTensor):  #
        return img.float().div(255)
    else:
        return img


def scale(img, mask, factor=-1):
    """
    按比例放大或缩小图像.
    :param img: color-RGB-[H,W,C]
    :param mask:grayscale-[H,W]
    :param factor: 缩放因子,值范围建议(不强制)在(0,2~]
           1为返回原图
    """
    if factor == 1:
        return img, mask
    elif factor <= 0:
        raise ValueError('scale_factor != 0')

    h, w = img.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    # 缩放图像
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    # padding or crop
    x, y = int(np.random.uniform(0, abs(new_w - w))), int(np.random.uniform(0, abs(new_h - h)))
    if factor < 1.0:
        # padding
        img_padding = np.zeros([h, w, 3], dtype=np.uint8)
        img_padding[y:y + new_h, x:x + new_w, :] = img
        mask_padding = np.zeros([h, w], dtype=np.uint8)
        mask_padding[y:y + new_h, x:x + new_w] = mask
        return img_padding, mask_padding
    elif factor > 1.0:
        # crop
        img_crop = img[y:y + h, x:x + w, :]
        mask_crop = mask[y:y + h, x:x + w]
        return img_crop, mask_crop


def adjust_brightness(img, factor):
    """
    调整一幅图像的亮度
    :param img:符合Opencv读入格式,unit8类型.
    :param factor:控制亮度强度.值范围应该在[0,2~].
           0为纯黑图 1为原图
    """
    lut = np.arange(0, 255 + 1) * factor
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    img = cv2.LUT(img, lut)
    return img


def adjust_contrast(img, factor):
    """
    调整一幅图像的对比度
    :param img:符合Opencv读入格式,unit8类型.
    :param factor:控制对比度强度.值范围应该在[0,2~].
           0为纯灰图 1为原图
    """
    lut = np.arange(0, 255 + 1) * factor
    if is_gray_img(img):
        mean = img.mean()
    else:
        mean = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean()
    lut = lut + (1 - factor) * mean
    lut = np.clip(lut, 0, 255).astype(img.dtype)
    img = cv2.LUT(img, lut)
    return img


def adjust_saturation(img, factor, gamma=0):
    """
    调整一幅图像的饱和度,label不做变换.
    :param img:符合Opencv读入格式,颜色通道为RGB,unit8类型.
    :param factor:控制饱和度强度.值范围应该在[0,2~].
           0为灰度图 1为原图
    """
    if factor == 1:
        return img

    if is_gray_img(img):
        gray = img
        return gray
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if factor == 0:
        return gray
    result = cv2.addWeighted(img, factor, gray, 1 - factor, gamma=gamma)
    result = np.clip(result, 0, 255).astype(img.dtype)
    return result


def ndarrayImageShow(image, window_name=None):
    """
    显示一副图像,输入图像符合opencv格式,且颜色通道为RGB.
    图像类型为np.uint8时,值范围[0,255].
    图像类型为np.float32时,值范围[0,1].
    对于彩色图像[H,W,3]
    对于灰度图像[H,W]
    """
    if window_name == None:
        window_name = str(time.time_ns())
    else:
        window_name = str(window_name)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    if image.ndim == 2:
        cv2.imshow(window_name, image)  # 灰度图
    else:
        cv2.imshow(window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def is_gray_img(img: np.ndarray):
    """
    用作所有功能函数中做类型检查判断。
    """
    return (len(img.shape) == 2) or (len(img.shape) == 3 and img.shape[-1] == 1)
