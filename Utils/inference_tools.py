# _*_ coding: utf-8 _*_
# @Time : 2022/4/9 22:16
# @Author : yc096
# @File : inference_tools.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import time
import os

import Config


class inference_tools():
    def __init__(self):
        self.demo_tensor = torch.randn([1, 1, 352, 352])

    def tensorShowGrayImage(self, tensor, window_name=None):
        '''
        显示单张灰度图,注意:执行show()方法才会真正显示出来
        tensor:经过Sigmoid激活过的特征组,格式应该为[1,1,H,W],即单张图像包含1个通道尺寸为H,W,默认类型为torch.float
        window_name:显示图像的窗体名
        '''
        grayImage = self.tensor2GrayImage(tensor)
        self.ndarrayImageShow(grayImage, window_name)

    def tensorShowRGBImage(self, tensor, window_name=None):
        '''
        显示单张彩色图,注意:执行show()方法才会真正显示出来
        tensor:经过Sigmoid激活过的特征组,格式应该为[1,1,H,W],即单张图像包含3个通道尺寸为H,W,默认类型为torch.float
        window_name:显示图像的窗体名
        '''
        rgbImage = self.tensor2RGBImage(tensor)
        self.ndarrayImageShow(rgbImage, window_name)

    def tensor2GrayImage(self, tensor):
        # tensor:经过Sigmoid激活过的特征组,格式应该为[1,1,H,W],即单张图像包含1个通道尺寸为H,W.默认类型为torch.float
        grayimage = np.squeeze(tensor.cpu().numpy(), axis=(0, 1))  # [B,C,H,W] ->[H,W]
        return grayimage

    def tensor2RGBImage(self, tensor):
        # tensor:经过Sigmoid激活过的特征组,格式应该为[1,3,H,W],即单张图像包含3个通道尺寸为H,W.默认类型为torch.float
        rgbImage = np.squeeze(tensor.cpu().numpy(), axis=0)  # [B,C,H,W] ->[C,H,W]
        rgbImage = np.transpose(rgbImage, [1, 2, 0])  # [C,H,W] -> [H,W,C]
        return rgbImage

    def showFeatures(self, tensor, numRows=3, numCols=5, filename='temp', resize_hw=None, is_colormap=True,RGB2BGR=False):
        '''
        用于显示一组特征,抽取所有通道中的特征组成一张大图.注意,该方法会将
        tensor:经过Sigmoid激活过的特征组,格式应该为[1,N,H,W],即单张图像包含N个通道尺寸为H,W.默认类型为torch.float
        numRows,numCols:控制大图的行列数
        filename:输出图像名
        resize_hw:是否将特征统一缩放到某一个尺度,用于防止特征组的尺度太小
        '''
        if resize_hw != None:
            tensor = F.interpolate(tensor, size=resize_hw, mode='bilinear', align_corners=True)
        b, c, h, w = tensor.cpu().numpy().shape
        features = np.squeeze(tensor.cpu().numpy(), axis=0)  # [B,C,H,W]  ->  [C,H,W]
        imgs = []  # 用于存放每个通道的特征图
        gap = 10  # 控制行列间距
        if is_colormap:
            col_gap  = np.zeros((h, gap, 3), np.uint8)  # 列间距图像
            row_gap  = np.zeros((gap, w * numCols + gap * (numCols - 1), 3), np.uint8)  # 行间距图像
        else:
            col_gap  = np.zeros((h, gap), np.uint8)  # 列间距图像
            row_gap  = np.zeros((gap, w * numCols + gap * (numCols - 1)), np.uint8)  # 行间距图像
        for i in range(c):
            img = features[i]
            img = (img * 255).astype(np.uint8)
            if is_colormap:
                img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 将单通道的灰度图转为伪彩色图
            imgs.append(img)
        img_rows = []  # 存储每一行拼接好的特征图
        for i in range(numRows):
            img_row = []
            for j in range(numCols):
                index = i * numCols + j  # 获取通道图像索引号
                if index >= c:
                    if is_colormap:
                        img = np.zeros((w, h, 3), np.uint8)  # 索引号越界,用纯黑图代替
                    else:
                        img = np.zeros((w, h, 1), np.uint8)
                else:
                    img = imgs[index]
                img_row.append(img)
                img_row.append(col_gap)  # 插入列间距图像
            img_row = np.hstack(img_row[:-1])  # 拼凑一行图像
            img_rows.append(img_row)
            img_rows.append(row_gap)  # 插入行间距图像
        features = np.vstack(img_rows[:-1])  # 将每一行图像,按垂直方向拼接
        self.imageWrite(features, filename,RGB2BGR=False)

    def tensorWrite(self, tensor, filename='temp',RGB2BGR=False):
        '''
        用于保存单张彩色图像或灰度图
        tensor:经过Sigmoid激活过的特征组,格式应该为[1,1or3,H,W],即单张彩色图像或者灰度图尺寸为H,W.默认类型为torch.float
        filename:输出图像名
        '''
        image = np.squeeze(tensor.cpu().numpy(), axis=0)  # [B,C,H,W] ->[C,H,W]
        if image.shape[0] == 1:
            image = np.squeeze(image, axis=0)  # [C, H, W] -> [H, W]
        else:
            image = np.transpose(image, [1, 2, 0])  # [C,H,W] -> [H,W,C]
        image = (image * 255).astype(np.uint8)
        self.imageWrite(image, filename,RGB2BGR)

    def imageWrite(self, image, filename=None,RGB2BGR=False):
        '''
        用于保存单张彩色图像或灰度图
        image:型为np.uint8时,值范围[0,255]
        filename:输出图像名
        注意:项目内读取彩色图像时,默认颜色通道是RGB,但为了调用opencv api需要将图像转为BGR通道
        '''
        if filename == None:
            filename = str(time.time_ns()) + '.jpg'
        path = os.path.join(Config.IMAGE_ROOT, filename+'.jpg')
        if image.ndim == 3 and RGB2BGR:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image)

    def ndarrayImageShow(self, image, window_name=None):
        """
        显示一副图像,输入图像符合opencv格式,且颜色通道为RGB
        图像类型为np.uint8时,值范围[0,255]
        图像类型为np.float32时,值范围[0,1]
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

    def show(self):
        cv2.waitKey()
