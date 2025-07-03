from model.nets.teacher import *
from model.nets.models import *
from model.nets.student import *

in_channels = [128, 256, 512, 1024, 2048]


class Teacher(nn.Module):
    def __init__(self, n_classes, in_channels_1, in_channels_2=5):
        super(Teacher, self).__init__()
        self.n_classes = n_classes

        self.in_channels_1 = in_channels_1
        self.in_channels_2 = in_channels_2

        self.backbone = resnet18(self.in_channels_1)

        self.s1 = RCB(in_channels[0])
        self.s2 = RCB(in_channels[1])
        self.s3 = RCB(in_channels[2])

        self.FC_Spatical_Channel = SCConv(64, 128)
        self.Edge_Refinement = Edge_Refinement(self.n_classes, self.in_channels_2)

    def forward(self, x):
        t1, t2, t3 = self.backbone(x)
        #print(t1.shape, t2.shape, t3.shape)

        t1 = self.s1(t1)
        t2 = self.s2(t2)
        t3 = self.s3(t3)

        t3 = F.upsample_bilinear(t3, t1.size()[2:])

        t = self.FC_Spatical_Channel(t1, t3)

        # print(s_out.shape)

        t_out = self.Edge_Refinement(t, t2, x)

        outputs = []
        outputs.append(t_out)

        return outputs


if __name__ == '__main__':
    x = torch.randn(1, 1, 256, 256)  # 创建随机输入张量
    model = Teacher(2, 1, 5)  # 创建 ScConv 模型
    print(model(x).shape)  # 打印模型输出的形状
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))