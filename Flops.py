import torch
from thop import profile
from model.LRNNet import *

device = torch.device("cpu")
# input_shape of model,batch_size=1
net = LRNNet(2, 1)  ##定义好的网络模型

input = torch.randn(1, 1, 256, 256)
flops, params = profile(net, inputs=(input,))

print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
print("params=", str(params / 1e6) + '{}'.format("M"))