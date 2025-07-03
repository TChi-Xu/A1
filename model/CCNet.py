import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from model.nets.student import *
from Net.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

import torch
import torch.nn as nn
import torch.nn.functional as F



class CCNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CCNet, self).__init__()
        self.backbone = resnet18(in_channels)
        self.decode_head = RCCAModule(recurrence=2, in_channels=512, num_classes=num_classes)

    def forward(self, x):
        outputs = []
        x1, x2, x3 = self.backbone(x)
        x = self.decode_head(x3)
        outputs.append(x)
        return  outputs

class CrissCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrissCrossAttention, self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // 8
        self.ConvQuery = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvKey = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvValue = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

        self.SoftMax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, _, h, w = x.size()

        # [b, c', h, w]
        query = self.ConvQuery(x)
        # [b, w, c', h] -> [b*w, c', h] -> [b*w, h, c']
        query_H = query.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)
        # [b, h, c', w] -> [b*h, c', w] -> [b*h, w, c']
        query_W = query.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)

        # [b, c', h, w]
        key = self.ConvKey(x)
        # [b, w, c', h] -> [b*w, c', h]
        key_H = key.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        # [b, h, c', w] -> [b*h, c', w]
        key_W = key.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        # [b, c, h, w]
        value = self.ConvValue(x)
        # [b, w, c, h] -> [b*w, c, h]
        value_H = value.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        # [b, h, c, w] -> [b*h, c, w]
        value_W = value.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        # [b*w, h, c']* [b*w, c', h] -> [b*w, h, h] -> [b, h, w, h]
        energy_H = (torch.bmm(query_H, key_H) + self.INF(b, h, w)).view(b, w, h, h).permute(0, 2, 1, 3)
        # [b*h, w, c']*[b*h, c', w] -> [b*h, w, w] -> [b, h, w, w]
        energy_W = torch.bmm(query_W, key_W).view(b, h, w, w)
        # [b, h, w, h+w]  concate channels in axis=3

        concate = self.SoftMax(torch.cat([energy_H, energy_W], 3))
        # [b, h, w, h] -> [b, w, h, h] -> [b*w, h, h]
        attention_H = concate[:, :, :, 0:h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h)
        attention_W = concate[:, :, :, h:h + w].contiguous().view(b * h, w, w)

        # [b*w, h, c]*[b*w, h, h] -> [b, w, c, h]
        out_H = torch.bmm(value_H, attention_H.permute(0, 2, 1)).view(b, w, -1, h).permute(0, 2, 3, 1)
        out_W = torch.bmm(value_W, attention_W.permute(0, 2, 1)).view(b, h, -1, w).permute(0, 2, 1, 3)

        return self.gamma * (out_H + out_W) + x

def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class RCCAModule(nn.Module):
    def __init__(self, recurrence=2, in_channels=2048, num_classes=33):
        super(RCCAModule, self).__init__()
        self.recurrence = recurrence
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.inter_channels = in_channels // 4
        self.conv_in = nn.Sequential(
            nn.Conv2d(self.in_channels, self.inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.inter_channels)
        )
        self.CCA = CrissCrossAttention(self.inter_channels)
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.inter_channels, self.inter_channels, 3, padding=1, bias=False)
        )
        self.cls_seg = nn.Sequential(
            nn.Conv2d(self.in_channels + self.inter_channels, self.inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.inter_channels),
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True),
            nn.Conv2d(self.inter_channels, self.num_classes, 1)
        )

    def forward(self, x):
        # reduce channels from C to C'   2048->512
        output = self.conv_in(x)

        for i in range(self.recurrence):
            output = self.CCA(output)

        output = self.conv_out(output)
        output = self.cls_seg(torch.cat([x, output], 1))
        return output


if __name__ == "__main__":
    model = CCNet(num_classes=2, in_channels=1)
    x = torch.randn(1, 1, 256, 256)
    model.cuda()
    out = model(x.cuda())
    print(out[0].shape)