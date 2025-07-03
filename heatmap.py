import os
import visdom
import torch
import cv2
from config import cfg
import argparse
import numpy as np
from PIL import Image
import torchvision.utils as vutils
from torchvision import transforms
from Models import get_segmentation_Model
# 新建一个连接客户端
# 指定env = 'test1'
vis = visdom.Visdom(env='test1')
# x = torch.arange(1, 100, 0.01)
# y = torch.sin(x)
# vis.line(X=x,Y=y, win='sinx',opts={'title':'y=sin(x)'})

# input_pic = 'data/test/images/301.png'

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
# model and dataset
parser.add_argument('--model', type=str, default='FCN_8',
                    choices=['FCN_8', 'FCN_16', 'FCN_32',
                             'U_Net'],
                    help='model name (default: fcn32s)')
parser.add_argument('--backbone', type=str, default='vgg16',
                    choices=['vgg16', 'resnet50'],
                    help='backbone name (default: vgg16)')
parser.add_argument('--dataset', type=str, default='potsdam',
                    choices=['voc2012', 'potsdam'],
                    help='dataset name (default: pascal_voc)')
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# output folder
best_filename = '{}_{}_{}_best_model.pth'.format(args.model, args.backbone, args.dataset)
best_weight_path = os.path.join(cfg.DATA.WEIGHTS_PATH, best_filename)
# image transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
input_pic = 'data/test/images/303.png'


model = get_segmentation_Model(name=args.model, nclass=cfg.TRAIN.CLASSES, pre_trained_base=False).to(device)
model_dict = torch.load(best_weight_path)
model.load_state_dict(model_dict)
print('Finished loading model!')

image = Image.open(input_pic)
images = transform(image).unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
    output = model(images)

    final_out = torch.zeros(512, 512)
    for i in range(6):
        out = torch.squeeze(output[0][:, i, :, :])

        print('out', out.shape)
        final_out += out
    print(final_out)
    final_out = torch.unsqueeze(final_out, 1)
    final_out = torch.unsqueeze(final_out, 0)
    kernel_grid = vutils.make_grid(final_out, normalize=True, scale_each=True)  # 1*输入通道数, w, h
    print(kernel_grid)
    kernel_grid = kernel_grid.numpy()

    print(kernel_grid)
    kernel_grid = kernel_grid * 255

    print(kernel_grid)

norm_img = np.zeros(kernel_grid.shape)
cv2.normalize(kernel_grid, norm_img, 0, 255, cv2.NORM_MINMAX)
norm_img = np.asarray(kernel_grid, dtype=np.uint8)

heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET) # 注意此处的三通道热力图是cv2专有的GBR排列
heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)# 将BGR图像转为RGB图像


cv2.namedWindow('heat')
cv2.imshow('heat', heat_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
