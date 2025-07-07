#!/usr/bin/python
# -*- coding: utf-8 -*-
from torch import nn
import torch
from torchvision.datasets import ImageFolder
from torchsummary import summary
from torch.nn import functional as F
#from attention import Non_Local
#from attention import *

class AutoEncoder(nn.Module):
    def __init__(self, out_channels=384):
        super(AutoEncoder, self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8)



        # decoder
        self.upsample1 = nn.Upsample(size=3, mode='bilinear')
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2)

        self.upsample2 = nn.Upsample(size=8, mode='bilinear')
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2)

        self.upsample3 = nn.Upsample(size=15, mode='bilinear')
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2)

        self.upsample4 = nn.Upsample(size=32, mode='bilinear')
        self.conv10 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=4, stride=1, padding=2)

        self.upsample5 = nn.Upsample(size=63, mode='bilinear')
        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2)

        self.upsample6 = nn.Upsample(size=127, mode='bilinear')
        self.conv12 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=2)

        self.upsample7 = nn.Upsample(size=128, mode='bilinear')
        self.conv13 = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv_final = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

        self.noise = FeatureNoise(noise_std=0.35)
    def forward(self, x):
        # Encoder
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))  # add attention
        x = self.conv6(x5)


        # Decoder
        x = self.upsample1(x)
        x = self.relu(self.conv7(x))
        x = self.noise(x)
        x = self.upsample2(x)
        x = self.relu(self.conv8(x))  # add attention
        x = self.noise(x)
        x = self.upsample3(x)
        x = self.relu(self.conv9(x))
        x = torch.cat([x5, x], dim=1)
        x = self.noise(x)
        x = self.upsample4(x)
        x = torch.cat([x4, x], dim=1)

        x = self.relu(self.conv10(x))
        x = self.noise(x)

        x = self.upsample5(x)

        x = self.relu(self.conv11(x))
        x = torch.cat([x3, x], dim=1)
        # x = self.dropout(x)
        x = self.upsample6(x)

        x = self.relu(self.conv12(x))
        x = self.noise(x)

        x = self.upsample7(x)
        x = torch.cat([x2, x], dim=1)
        x = self.relu(self.conv13(x))
        x = self.conv_final(x)
        return x


def get_autoencoder(out_channels=384):
    return nn.Sequential(
        # encoder
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8),
        # decoder
        nn.Upsample(size=3, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=8, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=15, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=32, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=63, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=127, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=128, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3,
                  stride=1, padding=1)
    )

#-----------------------------------------------------------------------------------------------------------------
# def autopad(k, p=None):
#    """用于Conv函数和Classify函数中
#    根据卷积核大小k自动计算卷积核padding数（0填充）
#    v5中只有两种卷积：
#       1、下采样卷积:conv3x3 s=2 p=k//2=1
#       2、feature size不变的卷积:conv1x1 s=1 p=k//2=1
#    :params k: 卷积核的kernel_size
#    :return p: 自动计算的需要pad值（0填充）
#    """
#    if p is None:
#       p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # 自动计算pad数
#    return p
class FeatureNoise(nn.Module):
    def __init__(self, noise_std=0.35):
        """
        参数：
          noise_std: 噪声标准差，即控制添加噪声的强度，默认值为 0.1
        """
        super(FeatureNoise, self).__init__()
        self.noise_std = noise_std

    def forward(self, x):
        # 如果处于训练模式，生成高斯噪声并加到输入特征上
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        else:
            return x
#
# class Conv(nn.Module):
#    # Standard convolution
#    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#       super(Conv, self).__init__()
#       self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
#       self.bn = nn.BatchNorm2d(c2)
#       self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#
#    def forward(self, x):
#       return self.act(self.bn(self.conv(x)))
#
#    def fuseforward(self, x):
#       return self.act(self.conv(x))
#
#
# class Focus(nn.Module):
#    # Focus wh information into c-space
#    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#       super(Focus, self).__init__()
#       self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
#       # self.contract = Contract(gain=2)
#
#    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
#       return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
#       # return self.conv(self.contract(x))
# ---------------------------------------------------------------------------------------------------------------------
class Pdn_small(nn.Module):
    def __init__(self, out_channels):
        super(Pdn_small, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1,dilation=1)
        self.relu1 = nn.LeakyReLU(inplace=True)


        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=2,dilation=2)
        self.relu2 = nn.LeakyReLU(inplace=True)


        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=4,dilation=4)
        self.relu3 = nn.LeakyReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4, padding=12,dilation=8)

    def forward(self, input):
        output = self.focus1(self.relu1(self.conv1(input)))
        output = self.focus2(self.relu2(self.conv2(output)))
        output = self.relu3(self.conv3(output))
        output = self.conv4(output)

        return output
# ---------------------------------------------------------------------------------------------------------------------
class Pdn_small2(nn.Module):
    def __init__(self, out_channels):
        super(Pdn_small2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=3)
        self.relu1 = nn.LeakyReLU(inplace=True)


        self.conv2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=3)
        self.relu2 = nn.LeakyReLU(inplace=True)


        self.conv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.relu3 = nn.LeakyReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)

    def forward(self, input):
        output = self.focus1(self.relu1(self.conv1(input)))
        output = self.focus2(self.relu2(self.conv2(output)))
        output = self.relu3(self.conv3(output))
        output = self.conv4(output)

        return output
# ------------------
# ---------------------------------------------------------------------------------------------------

# class FFE(nn.Module):
#     def __init__(self, in_channels, kernel_size=3):
#         """
#         Frequency-based Feature Enhancement (FFE) Module
#         :param in_channels: 输入通道数
#         :param kernel_size: 低通卷积核大小 (默认 3)
#         """
#         super(FFE, self).__init__()
#
#         # 低通卷积（模仿高斯滤波器）
#         self.low_pass_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=in_channels, bias=False)
#
#         # 线性映射（1x1卷积调整通道数）
#         self.linear_mapping = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
#
#     def forward(self, x):
#         """
#         前向传播过程
#         :param x: 输入特征图 (batch_size, in_channels, H, W)
#         :return: 增强后的特征图
#         """
#         # 低通滤波 (保留低频信息)
#         low_freq_features = self.low_pass_conv(x)
#
#         # 拼接原始特征与低频特征
#         fused_features = torch.cat([x, low_freq_features], dim=1)
#
#         # 线性映射 (调整通道数)
#         enhanced_features = self.linear_mapping(fused_features)
#
#         # 元素级相加 (融合原始特征)
#         output = enhanced_features + x
#
#         return output

class Residual(nn.Module):
    def __init__(self, input_channels, output_channels, use_11conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        if use_11conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class Pdn_small3(nn.Module):
    def __init__(self, out_channels):
        super(Pdn_small3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.res1 = Residual(16, 16, use_11conv=True)



        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.res2 = Residual(128, 128, use_11conv=True)


        self.conv3 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.res3 = Residual(128, 128, use_11conv=True)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, input):
        output = self.focus1(self.res1(self.relu1(self.conv1(input))))
        output = self.focus2(self.res2(self.relu2(self.conv2(output))))
        output = self.res3(self.relu3(self.conv3(output)))
        output = self.conv4(output)
        return output
# ---------------------------------------------------------------------------------------------------------------------
# class Residual(nn.Module):
#     def __init__(self, input_channels, output_channels, use_11conv=False, strides=1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=strides)
#         self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
#         if use_11conv:
#             self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=strides)
#         else:
#             self.conv3 = None
#         self.bn1 = nn.BatchNorm2d(output_channels)
#         self.bn2 = nn.BatchNorm2d(output_channels)
#
#     def forward(self, X):
#         Y = F.relu(self.bn1(self.conv1(X)))
#         Y = self.bn2(self.conv2(Y))
#         if self.conv3:
#             X = self.conv3(X)
#         Y += X
#         return F.relu(Y)
#
#
# class Pdn_small3(nn.Module):
#     def __init__(self, out_channels):
#         super(Pdn_small3, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
#         self.relu1 = nn.LeakyReLU(inplace=True)
#         self.res1 = Residual(16, 16, use_11conv=True)
#
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=128, kernel_size=3, padding=1)
#         self.relu2 = nn.LeakyReLU(inplace=True)
#         self.res2 = Residual(128, 128, use_11conv=True)
#
#         self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
#         self.relu3 = nn.LeakyReLU(inplace=True)
#         self.res3 = Residual(128, 128, use_11conv=True)
#
#         # FFE 模块
#         self.ffe = FFE(in_channels=128)
#
#         self.conv4 = nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=3, padding=1)
#
#     def forward(self, input):
#         output = self.res1(self.relu1(self.conv1(input)))
#         output = self.res2(self.relu2(self.conv2(output)))
#         output = self.res3(self.relu3(self.conv3(output)))
#
#         # 使用 FFE 模块增强低频特征
#         output = self.ffe(output)
#
#         output = self.conv4(output)
#         return output

class Small(nn.Module):
    def __init__(self, out_channels=384):
        super(Small, self).__init__()
        self.out_channels = out_channels
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, padding=3)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=3)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
        self.avgPool = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        x = self.avgPool(self.relu(self.conv1(x)))
        x = self.avgPool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x

class Medium(nn.Module):
    def __init__(self, out_channels=384):
        super(Medium, self).__init__()
        self.out_channels = out_channels
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, padding=3)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, padding=3)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4)
        self.conv6 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,  kernel_size=1)
        self.avgPool = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        x = self.avgPool(self.relu(self.conv1(x)))
        x = self.avgPool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)
        return x

def get_pdn_small(out_channels=384):
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, padding=3),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=3),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
    )

def get_pdn_medium(out_channels=384):
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, padding=3),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, padding=3),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,  kernel_size=1)
    )

class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample

class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path

def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)


# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# x = torch.rand((1, 3, 512, 512))
# net = AutoEncoder(out_channels=384)
# # x = torch.rand((1, 3, 640, 640))
# y = net(x)
# print(y.shape)
# # print(y.shape)
# summary(net, (3, 256, 256))