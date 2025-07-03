# -- coding: utf-8 --
from thop import profile
from model.FasterNet import *
from model.GCtx_UNet import *
# Model
print('==> Building model..')

model = FasterNet(nclass=2, in_channels=1)

dummy_input = torch.randn(1, 1, 256, 256)
model.cuda()
flops, params = profile(model, (dummy_input.cuda(),))
print('flops: %.2f GFLOPS, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
