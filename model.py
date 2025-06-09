import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import torch.nn.functional as F
from collections import OrderedDict



class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        return self.sigmoid(self.conv(out))


class CBAM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel_att = ChannelAttentionModule(channel)
        self.spatial_att = SpatialAttentionModule()

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DifferenceUNet_CBAM(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # 双分支编码器（共享权重）
        self.encoderA = nn.ModuleList([
            DoubleConv(in_channels, 64),
            DoubleConv(64, 128),
            DoubleConv(128, 256),
            DoubleConv(256, 512),
            DoubleConv(512, 1024)
        ])
        self.encoderB = nn.ModuleList([
            DoubleConv(in_channels, 64),
            DoubleConv(64, 128),
            DoubleConv(128, 256),
            DoubleConv(256, 512),
            DoubleConv(512, 1024)
        ])


        # 带CBAM的差异模块
        self.diff_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                CBAM(64),  # 添加CBAM
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1),
                CBAM(128),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                CBAM(256),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, 3, padding=1),
                CBAM(512),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(1024, 1024, 3, padding=1),
                CBAM(1024),
                nn.ReLU()
            )
        ])

        # 解码器
        self.decoder = nn.ModuleDict({
            'up0': self._up_block(1024, 512),
            'conv0': DoubleConv(1024, 512),
            'up1': self._up_block(512, 256),
            'conv1': DoubleConv(512, 256),
            'up2': self._up_block(256, 128),
            'conv2': DoubleConv(256, 128),
            'up3': self._up_block(128, 64),
            'conv3': DoubleConv(128, 64),
            'out': nn.Conv2d(64, out_channels, 1)
        })
        self.pool = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )

    def forward(self, x1, x2):
        diffs = []
        for i in range(5):
            x1 = self.encoderA[i](x1)
            x2 = self.encoderB[i](x2)

            # 计算差异特征
            diff = torch.abs(x1 - x2)

            # # 差异特征计算可试着改为拼接加卷积
            # diff = torch.cat([x1, x2], dim=1)

            diff = self.diff_blocks[i](diff)  # 经过CBAM处理
            diffs.append(diff)

            if i < 4:
                x1 = self.pool(x1)
                x2 = self.pool(x2)

        # 解码过程
        x = self.decoder['up0'](diffs[-1])
        x = torch.cat([x, diffs[3]], dim=1)
        x = self.decoder['conv0'](x)

        x = self.decoder['up1'](x)
        x = torch.cat([x, diffs[2]], dim=1)
        x = self.decoder['conv1'](x)

        x = self.decoder['up2'](x)
        x = torch.cat([x, diffs[1]], dim=1)
        x = self.decoder['conv2'](x)

        x = self.decoder['up3'](x)
        x = torch.cat([x, diffs[0]], dim=1)
        x = self.decoder['conv3'](x)

        return self.sigmoid(self.decoder['out'](x))