import torch.nn as nn
import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的函数库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块


class RCB(nn.Module):

    def __init__(self, features, out_features=64):
        super(RCB, self).__init__()

        self.conv = nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False)

        self.conv_up = nn.Sequential(
                                   # 深度可分离卷积
                                   nn.Conv2d(out_features, out_features*2, kernel_size=3, padding=1, groups= out_features),
                                   nn.BatchNorm2d(out_features*2),
                                   nn.ReLU6(inplace=True),
                                   nn.Conv2d(out_features * 2, out_features, kernel_size=1, padding=0, stride=1, groups=1),
                                   )

        self.conv_low = nn.Sequential(
            # 深度可分离卷积
            nn.Conv2d(out_features, out_features // 2, kernel_size=3, padding=1, groups=out_features//2),
            nn.BatchNorm2d(out_features // 2),
            nn.ReLU6(inplace=True),
            nn.Conv2d(out_features // 2, out_features, kernel_size=1, padding=0, stride=1, groups=1),
        )

    def forward(self, x):
        x = self.conv(x)
        out_up = self.conv_up(x)
        out_low = self.conv_low(x)
        out = out_up + out_low + x
        return out


if __name__ == '__main__':
        x = torch.randn(1, 128, 64, 64)  # 创建随机输入张量
        model = RCB(128)
        print(model(x).shape)
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
