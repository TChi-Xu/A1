from torchvision.models import resnet50, resnet101
from torchvision.models._utils import IntermediateLayerGetter
import torch
import torch.nn as nn
from model.nets.student import *


class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels

        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )

    def forward(self, x):
        out_puts = []
        for ppm in self:
            ppm_out = nn.functional.interpolate(ppm(x), size=x.size()[-2:], mode='bilinear', align_corners=True)
            out_puts.append(ppm_out)
        return out_puts


class PSPHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6], num_classes=3):
        super(PSPHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes) * self.out_channels, self.out_channels, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out


# 构建一个FCN分割头，用于计算辅助损失
class Aux_Head(nn.Module):
    def __init__(self, in_channels=1024, num_classes=3):
        super(Aux_Head, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.decode_head = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.in_channels // 2),
            nn.ReLU(),

            nn.Conv2d(self.in_channels // 2, self.in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.in_channels // 4),
            nn.ReLU(),

            nn.Conv2d(self.in_channels // 4, self.num_classes, kernel_size=3, padding=1),

        )

    def forward(self, x):
        return self.decode_head(x)


class PSPNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(PSPNet, self).__init__()
        self.num_classes = num_classes
        self.backbone = resnet18(in_channels)
        self.decoder = PSPHEAD(in_channels=512, out_channels=256, pool_sizes=[1, 2, 3, 6],
                               num_classes=self.num_classes)
        self.cls_seg = nn.Sequential(
            nn.Conv2d(256, self.num_classes, kernel_size=3, padding=1),
        )

    def forward(self, x):
        _, _, h, w = x.size()
        x1, x2, x3 = self.backbone(x)
        x = self.decoder(x3)
        x = self.cls_seg(x)
        x = nn.functional.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        outputs = []

        outputs.append(x)
        return outputs


if __name__ == "__main__":
    model = PSPNet(in_channels=1, num_classes=2)
    model.eval()
    image = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        output = model.forward(image)
    print(output[0].shape)