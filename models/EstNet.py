import torch
import torch.nn as nn
from torch.nn import init

import numpy as np
from models.model_utils import *

class EstNet(nn.Module):
    def __init__(self, in_c=6, num_para=2, height=256, width=256, batchNorm=True):
        super(EstNet, self).__init__()

        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm,in_c,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=3, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  256, stride=2)
        self.conv4_1 = conv(self.batchNorm, 256,  256)
        self.conv5   = conv(self.batchNorm, 256,  256, stride=2)
        self.full_connect = fully_con(self.batchNorm, int(height * width / 4), num_para)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.kaiming_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.kaiming_uniform_(m.weight)
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        out_conv1 = self.conv1(x)                           # x=1024
        out_conv2 = self.conv2(out_conv1)                   # 256
        out_conv3 = self.conv3_1(self.conv3(out_conv2))     # 128
        out_conv4 = self.conv4_1(self.conv4(out_conv3))     # 64
        out_conv5 = self.conv5(out_conv4)                   # 32
        out = self.full_connect(out_conv5.view(out_conv5.size(0), -1))
        return out