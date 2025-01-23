import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DDFE(nn.Module):
    def __init__(self):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self.Tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv_h = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(6),
            self.leaky_relu,
            nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(12),
            self.leaky_relu)
        self.conv_l = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(6),
            self.leaky_relu)
        self.conv_u = nn.Sequential(
            nn.ConvTranspose2d(6, 12, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            norm_layer(12),
            self.leaky_relu)
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(12, 6, kernel_size=3, stride=1, padding=1, bias=False, groups=3),
            norm_layer(6),
            self.leaky_relu,
            nn.Conv2d(6, 3, kernel_size=1, stride=1, padding=0, groups=3))

    def forward(self, x, trimap):
        _, _, h, w = x.size()
        x_d = F.interpolate(x, scale_factor=1 / 2, mode='nearest')
        x_l = self.conv_l(x_d)
        x_l = self.conv_u(x_l)
        x_hf = self.conv_h(x)
        x_hf = self.conv_1_1(x_hf - x_l)
        x_hf = self.Tanh(x_hf)
        mask = (trimap == 0) | (trimap == 2)
        x_hf = x_hf.masked_fill(mask, 0)
        return x_d, x_hf


class Refinement(nn.Module):
    def __init__(self):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv_u = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1, bias=False, groups=3),
            norm_layer(6),
            self.leaky_relu,
            nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1, groups=3))
        self.conv_u2 = nn.Sequential(
            nn.Conv2d(12, 6, kernel_size=3, stride=1, padding=1, bias=False, groups=3),
            norm_layer(6),
            self.leaky_relu,
            nn.Conv2d(6, 3, kernel_size=1, stride=1, padding=0, groups=3))

    def forward(self, x, x_hf):
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        x = torch.add(x, x_hf)
        x = self.conv_u(x)
        x = self.conv_u2(x)
        x = (torch.tanh(x) + 1.0) / 2.0
        return x
