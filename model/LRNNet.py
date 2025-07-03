import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def Split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2


def Merge(x1, x2):
    return torch.cat((x1, x2), 1)


def Channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class FCB(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super(FCB, self).__init__()
        self.oup_inc = chann // 2
        self.chann = chann-self.oup_inc

        # dw
        self.conv3x1_1_l = nn.Conv2d(self.oup_inc, self.oup_inc, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1_l = nn.Conv2d(self.oup_inc, self.oup_inc, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1_l = nn.BatchNorm2d(self.oup_inc, eps=1e-03)

        # dw
        self.conv3x1_1_r = nn.Conv2d(self.oup_inc, self.oup_inc, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1_r = nn.Conv2d(self.oup_inc, self.oup_inc, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1_r = nn.BatchNorm2d(self.oup_inc, eps=1e-03)

        self.relu = nn.ReLU(inplace=True)
        # self.channel_shuffle = PermutationBlock(2)

        self.conv = nn.Sequential(
            nn.Conv2d(chann, chann, 3, stride=1, padding=1, bias=True),
            nn.Dropout2d(dropprob),
            nn.Conv2d(chann, chann, 3, stride=1, padding=dilated, bias=True, dilation=dilated),
            nn.Conv2d(chann, chann, 1, stride=1, bias=True),
            nn.BatchNorm2d(chann, eps=1e-03),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        residual = x
        x1, x2 = Split(x)

        output1 = self.conv3x1_1_l(x1)
        output1 = self.bn1_l(output1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_1_l(output1)
        output1 = self.bn1_l(output1)
        output1_mid = self.relu(output1)

        output2 = self.conv1x3_1_r(x2)
        output2 = self.bn1_r(output2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_1_r(output2)
        output2 = self.bn1_r(output2)
        output2_mid = self.relu(output2)

        out = Merge(output1_mid, output2_mid)
        out = self.conv(out)

        out = F.relu(residual + out)
        #print(out.size())
        # out = self.channel_shuffle(out)   ### channel shuffle
        out = Channel_shuffle(out, 2)


        return out
        # return    ### channel shuffle


class LRNNet(nn.Module):
    def __init__(self, nclass, in_channels):
        super(LRNNet, self).__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128, eps=1e-03),
            nn.ReLU(inplace=True),
        )
        self.FCB1 = FCB(128, dropprob=0.5, dilated=2)

        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256, eps=1e-03),
            nn.ReLU(inplace=True)
        )
        self.FCB2 = FCB(256, dropprob=0.5, dilated=2)

        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(512, eps=1e-03),
            nn.ReLU(inplace=True)
        )
        self.FCB3 = FCB(512, dropprob=0.5, dilated=2)

        self.up1_1 = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=True),
            nn.BatchNorm2d(64, eps=1e-03),
            nn.ReLU(inplace=True),

        )
        self.up1_2 = nn.Sequential(
            nn.Conv2d(64, 128, 1, bias=True),
            nn.BatchNorm2d(128, eps=1e-03),
            nn.ReLU(inplace=True),
        )

        self.classifier1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 1, bias=True),
            nn.BatchNorm2d(64, eps=1e-03),
            nn.ReLU(inplace=True)
        )

        self.up2_1 = nn.Sequential(
            nn.Conv2d(256, 64, 1, bias=True),
            nn.BatchNorm2d(64, eps=1e-03),
            nn.ReLU(inplace=True),

        )
        self.up2_2 = nn.Sequential(
            nn.Conv2d(64, 256, 1, bias=True),
            nn.BatchNorm2d(256, eps=1e-03),
            nn.ReLU(inplace=True),
        )

        self.classifier2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 1, bias=True),
            nn.BatchNorm2d(128, eps=1e-03),
            nn.ReLU(inplace=True)
        )

        self.up3_1 = nn.Sequential(
            nn.Conv2d(512, 64, 1, bias=True),
            nn.BatchNorm2d(64, eps=1e-03),
            nn.ReLU(inplace=True),

        )
        self.up3_2 = nn.Sequential(
            nn.Conv2d(64, 512, 1, bias=True),
            nn.BatchNorm2d(512, eps=1e-03),
            nn.ReLU(inplace=True),
        )

        self.classifier3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 1, bias=True),
            nn.BatchNorm2d(256, eps=1e-03),
            nn.ReLU(inplace=True)
        )

        self.act = nn.Sigmoid()

        self.conv = nn.Sequential(
            nn.Conv2d(64, nclass, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(nclass),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.down1(x)

        x1 = self.FCB1(x1)
        x1 = self.FCB1(x1)
        x1 = self.FCB1(x1)

        x2 = self.down2(x1)
        x2 = self.FCB2(x2)
        x2 = self.FCB2(x2)

        x3 = self.down3(x2)
        x3 = self.FCB3(x3)
        x3 = self.FCB3(x3)
        x3 = self.FCB3(x3)
        x3 = self.FCB3(x3)
        x3 = self.FCB3(x3)
        x3 = self.FCB3(x3)
        x3 = self.FCB3(x3)
        x3 = self.FCB3(x3)

        x3_1 = self.up3_1(x3)
        x3_1_s = self.act(x3_1)
        x3_1 = x3_1 * x3_1_s
        x3_1 = self.up3_2(x3_1)
        x3 = self.classifier3(x3_1)

        x2 = x3+x2
        x2_1 = self.up2_1(x2)
        x2_1_s = self.act(x2_1)
        x2_1 = x2_1 * x2_1_s
        x2_1 = self.up2_2(x2_1)
        x2 = self.classifier2(x2_1)

        x1 = x1 + x2
        x1_1 = self.up1_1(x1)
        x1_1_s = self.act(x1_1)
        x1_1 = x1_1 * x1_1_s
        x1_1 = self.up1_2(x1_1)
        x1 = self.classifier1(x1_1)

        out = self.conv(x1)

        output = []
        output.append(out)

        return output


if __name__ == "__main__":
    model = LRNNet(nclass=2, in_channels=1)
    model.eval()
    image = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        output = model.forward(image)
    print(output[0].shape)