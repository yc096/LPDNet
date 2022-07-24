# _*_ coding: utf-8 _*_
# @Time : 2022/4/5 9:23 
# @Author : yc096
# @File : opencv_transform.py
import cv2
import numpy as np
import random
from Utils.opencv_function import ndarrayImageToTensor, scale, adjust_contrast, adjust_saturation, adjust_brightness


class ToTensor(object):
    def __call__(self, img_mask):
        img = img_mask[0]
        mask = img_mask[1]
        img = ndarrayImageToTensor(img)
        mask = ndarrayImageToTensor(mask)
        return img, mask

class CopyPaste(object):
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,img_mask):
        img1 = img_mask[0]
        mask1 = img_mask[1]
        img2 = img_mask[0]
        mask2 = img_mask[1]

        # 对两组图像随机水平反转
        if random.random()>self.p:
            img1 = self.flip(img1)
            mask1 = self.flip(mask1)
        if random.random()>self.p:
            img2 = self.flip(img2)
            mask2 = self.flip(mask2)
        # 对两组图像随机尺度变换
        if random.random() > self.p:
            img1, mask1 = self.large_scale_jittering(img1, mask1)
        if random.random() > self.p:
            img2, mask2 = self.large_scale_jittering(img2, mask2)

        image = self.img_add(img1, img2, mask2)
        mask = self.img_add(mask1, mask2, mask2)

        return image,mask
    def flip(self,img, mode=1):
        """
        mode: 0垂直翻转 1水平翻转 -1垂直水平翻转
        """
        return cv2.flip(img, mode)
    def flip(self,img, mode=1):
        """
        mode: 0垂直翻转 1水平翻转 -1垂直水平翻转
        """
        return cv2.flip(img, mode)
    def large_scale_jittering(self,img, mask, min_scale=0.25, max_scale=1.75):

        # 获取尺度信息
        scale_ratio = np.random.uniform(min_scale, max_scale)  # 随机获取变换尺度
        h, w = img.shape[:2]  # 获取原始图像高宽
        new_h, new_w = int(h * scale_ratio), int(w * scale_ratio)  # 获得新的图像高宽
        if scale_ratio == 1:
            return img, mask

        # 缩放图像
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # padding or crop
        x, y = int(np.random.uniform(0, abs(new_w - w))), int(np.random.uniform(0, abs(new_h - h)))

        if scale_ratio < 1.0:
            # padding
            img_padding = np.zeros([h, w, 3], dtype=np.uint8)
            mask_padding = np.zeros([h, w], dtype=np.uint8)
            img_padding[y:y + new_h, x:x + new_w, :] = img
            mask_padding[y:y + new_h, x:x + new_w] = mask
            return (img_padding, mask_padding)
        elif scale_ratio > 1.0:
            # crop
            img_crop = img[y:y + h, x:x + w, :]
            mask_crop = mask[y:y + h, x:x + w]
            return (img_crop, mask_crop)
    def img_add(self,img1, img2, mask2):
        """
        :param img1:
        :param img2:
        :param mask2:
        :return:
        """
        h, w = img1.shape[:2]  # 获取主图像的高宽

        mask = np.asarray(mask2, dtype=np.uint8)
        sub = cv2.add(img2, np.zeros(img2.shape, dtype=np.uint8), mask=mask)  # 裁剪出辅助图像的区域

        mask02 = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)  # 辅助图像的mask映射到主图像上
        sub2 = cv2.add(img1, np.zeros(img1.shape, dtype=np.uint8), mask=mask02)  #

        img1 = img1 - sub2

        img1 = img1 + cv2.resize(sub, (w, h), interpolation=cv2.INTER_NEAREST)
        return img1

class RandomHorizontalFlip(object):
    '''
    随机水平反转img和mask
    self.flip_mode : 0垂直翻转 1水平翻转 -1垂直水平翻转
    '''

    def __init__(self, p=0.5):
        self.p = p
        self.flip_mode = 1

    def __call__(self, img_mask):
        if random.random() > self.p:
            return img_mask
        else:
            img = img_mask[0]
            mask = img_mask[1]
            img = cv2.flip(img, self.flip_mode)
            mask = cv2.flip(mask, self.flip_mode)
            return img, mask


class RandomScale(object):
    def __init__(self, scale_retes=[0.75, 1, 1.25]):
        self.scale_rates = scale_retes

    def __call__(self, img_mask):
        rates = random.choice(self.scale_rates)
        img = img_mask[0]
        mask = img_mask[1]
        img, mask = scale(img, mask, factor=rates)
        return img, mask


class ColorJitter(object):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5):
        self.brightness = random.uniform(max(0, 1 - brightness), 1 + brightness)
        self.contrast = random.uniform(max(0, 1 - contrast), 1 + contrast)
        self.saturation = random.uniform(max(0, 1 - saturation), 1 + saturation)

    def __call__(self, img_mask):
        img = img_mask[0]
        img = adjust_brightness(img, self.brightness)
        img = adjust_contrast(img, self.contrast)
        img = adjust_saturation(img, self.saturation)
        return (img, img_mask[1])


class Resize(object):
    def __init__(self, resize=(256, 256)):
        self.resize_h, self.resize_w = resize

    def __call__(self, img_mask):
        img = img_mask[0]
        mask = img_mask[1]
        img = cv2.resize(img, (self.resize_w, self.resize_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.resize_w, self.resize_h), interpolation=cv2.INTER_NEAREST)
        return img, mask
