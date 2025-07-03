import numpy as np
from torch import nn, einsum
import torch
def get_readout_open(vit_features, features, start_index=1):
    readout = [
        ProjectReadout(vit_features, start_index) for out_feat in features
    ]
    return readout

class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index
        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)

        return self.project(features)

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1).contiguous()
        return x


class Read(nn.Module):
    def __init__(self,  vit_feature=1024, features=[128, 256, 512], size=[256, 256]):
        super().__init__()
        read_open = get_readout_open(vit_feature, features, start_index=1)

        self.change_channels1 =nn.Sequential(
            read_open[0],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d( in_channels=vit_feature, out_channels=features[0],kernel_size=1,stride=1,padding=0),
            nn.ConvTranspose2d(features[0], features[0],kernel_size=4,stride=4,padding=0, bias=True,dilation=1,groups =1)
        )
        self.change_channels2 =nn.Sequential(
              read_open[1],
              Transpose(1,2),
              nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
              nn.Conv2d( in_channels=vit_feature, out_channels=features[1], kernel_size=1,stride=1, padding=0),
              nn.ConvTranspose2d(features[1], features[1], kernel_size=2, stride=2, padding=0, bias=True,dilation=1, groups=1)
        )
        self.change_channels3 =nn.Sequential(
            read_open[2],
            Transpose(1,2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1]//16])),
            nn.Conv2d( in_channels=vit_feature, out_channels=features[2], kernel_size=1,stride=1, padding=0)
        )

        self.unflatten = nn.Sequential(
            nn.Unflatten(2, torch.Size([ size[0]// 16, size[1] // 16,]))
        )

    def forward(self, x1, x2, x3):

        x1 = self.change_channels1[0:2](x1)
        x2 = self.change_channels2[0:2](x2)
        x3 = self.change_channels3[0:2](x3)

        if x1.ndim == 3:
            x1 = self.unflatten(x1)
        if x2.ndim ==3:
            x2 = self.unflatten(x2)
        if x3.ndim ==3:
            x3 = self.unflatten(x3)

        x1 = self.change_channels1[3: len(self.change_channels1)](x1)
        x2 = self.change_channels2[3: len(self.change_channels2)](x2)
        x3 = self.change_channels3[3: len(self.change_channels3)](x3)

        return x1, x2, x3

if __name__ == '__main__':
    net = Read()
    print(net)





