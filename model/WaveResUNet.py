import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from model.nets.student import *
from Net.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class WaveBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4, qkv_bias=False, attn_drop=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, mode='fc'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PATM(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop, mode=mode)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PATM(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., mode='fc'):
        super().__init__()

        self.fc_h = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_c = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)

        self.tfc_h = nn.Conv2d(2 * dim, dim, (1, 7), stride=1, padding=(0, 7 // 2), groups=dim, bias=False)
        self.tfc_w = nn.Conv2d(2 * dim, dim, (7, 1), stride=1, padding=(7 // 2, 0), groups=dim, bias=False)
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1, 1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode

        if mode == 'fc':
            self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
            self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
        else:
            self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
                                              nn.BatchNorm2d(dim), nn.ReLU())
            self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
                                              nn.BatchNorm2d(dim), nn.ReLU())

    def forward(self, x):

        B, C, H, W = x.shape
        theta_h = self.theta_h_conv(x)
        theta_w = self.theta_w_conv(x)

        x_h = self.fc_h(x)
        x_w = self.fc_w(x)
        x_h = torch.cat([x_h * torch.cos(theta_h), x_h * torch.sin(theta_h)], dim=1)
        x_w = torch.cat([x_w * torch.cos(theta_w), x_w * torch.sin(theta_w)], dim=1)

        #         x_1=self.fc_h(x)
        #         x_2=self.fc_w(x)
        #         x_h=torch.cat([x_1*torch.cos(theta_h),x_2*torch.sin(theta_h)],dim=1)
        #         x_w=torch.cat([x_1*torch.cos(theta_w),x_2*torch.sin(theta_w)],dim=1)

        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        c = self.fc_c(x)
        a = F.adaptive_avg_pool2d(h + w + c, output_size=1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WaveResUNet(nn.Module):
    def __init__(self, nclass, in_channels):
        super(WaveResUNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        )

        self.res_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64)
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        )
        self.res_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        )
        self.res_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        )
        self.res_conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.WaveNetblock1 = WaveBlock(64)
        self.WaveNetblock2 = WaveBlock(128)
        self.WaveNetblock3 = WaveBlock(128)
        self.WaveNetblock4 = WaveBlock(32)


        self.res_0 = nn.Conv2d(64, 2,1,1)

        self.res_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        )

        self.res_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        )

        self.res_3 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        )


    def forward(self, input):
        x1 = self.conv1(input)
        y1 = self.res_conv_1(input)

        x2 = self.conv2(x1)
        y2 = self.res_conv_2(y1)

        x3 = self.conv3(x2)
        y3 = self.res_conv_3(y2)

        x4 = self.conv4(x3)
        y4 = self.res_conv_4(y3)

        y1 = self.WaveNetblock1(y1)
        y2 = self.WaveNetblock2(y2)
        y3 = self.WaveNetblock3(y3)
        y4 = self.WaveNetblock4(y4)

        x4 = F.upsample_bilinear(x4, x3.size()[2:])
        x3 = torch.cat([x4, y3], dim=1)
        x3 = self.res_3(x3)


        x3 = F.upsample_bilinear(x3, x2.size()[2:])
        x2 = torch.cat([x3, y2], dim=1)
        x2 = self.res_2(x2)


        x2 = F.upsample_bilinear(x2, x1.size()[2:])
        x1 = torch.cat([x2, y1], dim=1)
        x1 = self.res_1(x1)
        x = self.res_0(x1)

        outputs = []

        outputs.append(x)

        return outputs


if __name__ == "__main__":
    model = WaveResUNet(nclass=2, in_channels=3)
    model.eval()
    image = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model.forward(image)
    print(output[0].shape)
