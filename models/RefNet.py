import torch
import torch.nn as nn
from torch.nn import init

import numpy as np
from models.model_utils import *

class RefNet(nn.Module):
    def __init__(self, in_c=6, output_channels=6, batchNorm=True):
        super(RefNet, self).__init__()

        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm,  in_c,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        
        self.deconv4 = deconv(512, 256)
        self.deconv3 = deconv(768, 128)
        self.deconv2 = deconv(384, 64)
        self.deconv1 = deconv(192, 32)
        self.upsampled_flow = nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False)
        self.predict = nn.Conv2d(48, output_channels, kernel_size=3, stride=1, padding=1, bias=True)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.kaiming_normal_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.kaiming_normal_(m.weight)

    def forward(self, x):
        out_conv1 = self.conv1(x)                           # x=1024
        out_conv2 = self.conv2(out_conv1)                   # 256
        out_conv3 = self.conv3_1(self.conv3(out_conv2))     # 128
        out_conv4 = self.conv4_1(self.conv4(out_conv3))     # 64
        out_conv5 = self.conv5_1(self.conv5(out_conv4))     # 32
        # out_conv6 = self.conv6_1(self.conv6(out_conv5))     # 16

        out_deconv4 = self.deconv4(out_conv5)                 # 128
        concat4 = torch.cat((out_conv4, out_deconv4), 1)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3), 1)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        flow1 = self.upsampled_flow(concat1)
        r_b = self.predict(flow1)
        return r_b