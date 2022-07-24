# _*_ coding: utf-8 _*_
# @Time : 2022/5/25 15:11 
# @Author : yc096
# @File : LPDNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitialBlock, self).__init__()
        self.maxpool_branch = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels - 3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels - 3),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        conv_branch = self.conv_branch(x)
        maxpool_branch = self.maxpool_branch(x)

        out = torch.cat([conv_branch, maxpool_branch], dim=1)
        out = self.activation(out)
        return out


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, internal_ratio=4):
        super(DownsampleBlock, self).__init__()
        internal_channels = in_channels // internal_ratio
        self.activation = nn.ReLU
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=internal_channels),
            self.activation(),
            nn.Conv2d(internal_channels, internal_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=internal_channels),
            self.activation(),
            nn.Conv2d(internal_channels, in_channels * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=in_channels * 2)
        )

    def forward(self, x):
        conv_branch = self.conv(x)
        maxpool_branch = self.maxpool(x)
        maxpool_branch = torch.cat([maxpool_branch, maxpool_branch], dim=1)
        out = conv_branch + maxpool_branch
        out = F.relu(out)
        return out


class MSConvBlock(nn.Module):
    def __init__(self, in_out_channels, internal_ratio=3, branch_num=3, kernel_size=[3, 3, 3],
                 stride=[1, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1], dropout_prob=0.0):
        super(MSConvBlock, self).__init__()
        internal_channels = in_out_channels // internal_ratio
        self.spili_channel = internal_channels // branch_num
        self.activation = nn.ReLU
        self.residual_activation = self.activation()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_out_channels, internal_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=internal_channels),
            self.activation()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, in_out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=in_out_channels)
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(self.spili_channel, self.spili_channel, kernel_size=kernel_size[0], stride=stride[0],
                      padding=padding[0], dilation=dilation[0], bias=False),
            nn.BatchNorm2d(num_features=self.spili_channel),
            self.activation()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(self.spili_channel, self.spili_channel, kernel_size=kernel_size[1], stride=stride[1],
                      padding=padding[1], dilation=dilation[1], bias=False),
            nn.BatchNorm2d(num_features=self.spili_channel),
            self.activation()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(self.spili_channel, self.spili_channel, kernel_size=kernel_size[2], stride=stride[1],
                      padding=padding[2], dilation=dilation[2], bias=False),
            nn.BatchNorm2d(num_features=self.spili_channel),
            self.activation()
        )
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        residual = x
        b = self.conv1(x)
        b1, b2, b3 = torch.split(b, self.spili_channel, dim=1)
        b1 = self.branch1(b1)
        b2 = self.branch2(b2 + b1)
        b3 = self.branch3(b3 + b2 + b1)
        b = torch.cat([b1, b2, b3], dim=1)
        out = self.conv2(b)
        if self.dropout.p != 0:
            out = self.dropout(out)
        out = self.residual_activation(out + residual)
        return out


class LocalFeatureAggregation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalFeatureAggregation, self).__init__()
        self.activation = nn.ReLU
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation()
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation()
        )
        self.conv22 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv21 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation()
        )

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x12 = self.conv12(x1)
        x21 = self.conv21(x2)
        x1 = x1 + x21
        x2 = x2 + x12
        x1 = self.conv11(x1)
        x2 = self.conv22(x2)
        out = x1 + x2
        return out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x_sigmoid = self.activation(x)
        score = torch.clamp(x_sigmoid - 0.5, min=0, max=0.5)
        score = 1 - (score / 0.5)
        return score


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        avg = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg + max)
        return out


class Detail(nn.Module):
    def __init__(self, high_feature_channels, low_feature_channels):
        super(Detail, self).__init__()
        self.activation = nn.ReLU
        self.ca = ChannelAttention(high_feature_channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(low_feature_channels, high_feature_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(high_feature_channels),
            self.activation()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(high_feature_channels, high_feature_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(high_feature_channels),
            self.activation()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(high_feature_channels, high_feature_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(high_feature_channels),
            self.activation()
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x2 = self.conv1(x2)
        residual = x1
        out = x1 * x2
        out = out * self.ca(out)
        out = residual + out
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv2(out)
        out = self.conv3(out)
        return out


class LPDNet(nn.Module):
    def __init__(self):
        super(LPDNet, self).__init__()
        # init
        self.initialBlock_1 = InitialBlock(3, 54)  # H/2 , W/2
        # stage 1
        self.DownsampleBlock_2 = DownsampleBlock(54)  # H/4 , W/4
        self.ConvBlock_3 = MSConvBlock(108, dropout_prob=0.1)
        self.ConvBlock_4 = MSConvBlock(108, dropout_prob=0.1)
        self.ConvBlock_5 = MSConvBlock(108, dropout_prob=0.1)
        self.ConvBlock_6 = MSConvBlock(108, dropout_prob=0.1)
        # stage 2
        self.DownsampleBlock_7 = DownsampleBlock(108)  # H/8 , W/8
        self.ConvBlock_8 = MSConvBlock(216)
        self.ConvBlock_9 = MSConvBlock(216, padding=[1, 2, 4], dilation=[1, 2, 4], dropout_prob=0.2)
        self.ConvBlock_10 = MSConvBlock(216, padding=[2, 4, 8], dilation=[2, 4, 8], dropout_prob=0.2)
        self.ConvBlock_11 = MSConvBlock(216, padding=[1, 2, 4], dilation=[1, 2, 4], dropout_prob=0.2)
        # stage 3
        self.ConvBlock_12 = MSConvBlock(216)
        self.ConvBlock_13 = MSConvBlock(216, padding=[1, 2, 4], dilation=[1, 2, 4], dropout_prob=0.3)
        self.ConvBlock_14 = MSConvBlock(216, padding=[2, 4, 8], dilation=[2, 4, 8], dropout_prob=0.3)
        self.ConvBlock_15 = MSConvBlock(216, padding=[1, 2, 4], dilation=[1, 2, 4], dropout_prob=0.3)
        # stage 4
        self.ConvBlock_16 = MSConvBlock(216)
        self.ConvBlock_17 = MSConvBlock(216, padding=[1, 2, 4], dilation=[1, 2, 4], dropout_prob=0.3)
        self.ConvBlock_18 = MSConvBlock(216, padding=[2, 4, 8], dilation=[2, 4, 8], dropout_prob=0.3)
        self.ConvBlock_19 = MSConvBlock(216, padding=[1, 2, 4], dilation=[1, 2, 4], dropout_prob=0.3)
        # Cross-stage Features Aggregation
        self.LFA43 = LocalFeatureAggregation(216, 54)
        self.LFA32 = LocalFeatureAggregation(216, 54)
        self.LFA432 = LocalFeatureAggregation(54, 54)
        self.SpatialAttention = SpatialAttention()
        # -----Detail-----
        self.Detail = Detail(54, 108)
        self.pred_conv = nn.ConvTranspose2d(54, 1, kernel_size=3, padding=1, stride=2, output_padding=1, bias=False)

    def forward(self, x):
        init = self.initialBlock_1(x)
        stage1 = self.ConvBlock_6(self.ConvBlock_5(self.ConvBlock_4(self.ConvBlock_3(self.DownsampleBlock_2(init)))))
        stage2 = self.ConvBlock_11(self.ConvBlock_10(self.ConvBlock_9(self.ConvBlock_8(self.DownsampleBlock_7(stage1)))))
        stage3 = self.ConvBlock_15(self.ConvBlock_14(self.ConvBlock_13(self.ConvBlock_12(stage2))))
        stage4 = self.ConvBlock_19(self.ConvBlock_18(self.ConvBlock_17(self.ConvBlock_16(stage3))))
        LFA32 = self.LFA32(stage3, stage2)
        LFA32_ = LFA32 + LFA32 * self.SpatialAttention(LFA32)
        LFA43 = self.LFA43(stage4, stage3)
        LFA43_ = LFA43 + LFA43 * self.SpatialAttention(LFA43)
        LFA432 = self.LFA432(LFA43_, LFA32_)
        LFA432_ = LFA432 + LFA432 * self.SpatialAttention(LFA432)
        DETAIL = self.Detail(LFA432_, stage1)
        out = self.pred_conv(DETAIL)
        return out
