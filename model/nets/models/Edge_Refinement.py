import torch
from torch import nn, einsum
from model.nets.models.RCB import *



class Edge_Generation(nn.Module):
    def __init__(self, out_channels_1=64):
        super(Edge_Generation, self).__init__()
        self.conv2 = nn.Conv2d(out_channels_1, out_channels=2, kernel_size=1, padding=0, dilation=1, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x, y):
        out = x+y
        out = self.conv2(out)
        out_coarse = torch.cat([torch.mean(out, 1, keepdim=True), torch.max(out, 1, keepdim=True)[0]], 1)
        out_coarse = out * self.act(out_coarse)

        out_thin = torch.cat([torch.mean(out, 1, keepdim=True), torch.min(out, 1, keepdim=True)[0]], 1)
        out_thin = out * self.act(out_thin)

        return out_coarse, out_thin


class Edge_Refinement(nn.Module):
    def __init__(self, n_classes, in_channels_2=5, out_channels=64):
        super(Edge_Refinement, self).__init__()
        self.EdgeRefinement = Edge_Generation(64)

        self.conv0 = nn.Sequential(
            RCB(in_channels_2),
            nn.Conv2d(out_channels, n_classes, kernel_size=1, padding=0, dilation=1, bias=False)
        )

    def forward(self, x, y, raw):
        y = F.upsample_bilinear(y, x.size()[2:])

        out_coarse, out_thin = self.EdgeRefinement(x, y)
        out_coarse = F.upsample_bilinear(out_coarse, raw.size()[2:])
        out_thin = F.upsample_bilinear(out_thin, raw.size()[2:])
        # print(out_coarse.shape, out_thin.shape)

        out = self.conv0(torch.cat((out_coarse, out_thin, raw),1))
        return  out


if __name__ == '__main__':
    x = torch.randn(1, 64, 64, 64)
    y = torch.randn(1, 64, 32, 32)
    raw = torch.randn(1, 1, 256, 256) # 创建随机输入张量
    model = Edge_Refinement(2, 5)  # 创建 ScConv 模型
    print(model(x, y, raw).shape)  # 打印模型输出的形状
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))


