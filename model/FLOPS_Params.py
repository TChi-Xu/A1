# -- coding: utf-8 --
from thop import profile
from model.PP_LiteSeg import *
from model.FasterNet import *
# Model
print('==> Building model..')

model = FasterNet(2, 3)

dummy_input = torch.randn(1, 3, 256, 256)
model.cuda()
flops, params = profile(model, (dummy_input.cuda(),))
print('flops: %.2f GFLOPS, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
