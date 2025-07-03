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
params = model.state_dict()
v = params['score_pool4.weight']
c_int = v.size()[1]  # 输入层通道数
c_out = v.size()[0]  # 输出层通道数
print(c_int, c_out)
# for j in range(c_out):
#     kernel_j = v[j, :, :, :].unsqueeze(1)  # 压缩维度，为make_grid制作输入
#     kernel_grid = vutils.make_grid(kernel_j, normalize=True, scale_each=True, nrow=c_int)  # 1*输入通道数, w, h

# k_w, k_h = v.size()[-1], v.size()[-2]
# kernel_all = v.view(-1, 1, k_w, k_h)
# kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=c_int)  # 1*输入通道数, w, h
# vis.images(kernel_grid)
# for k, v in params.items():
#     print(k, params[k].shape)
image = Image.open(input_pic)
img1 = cv2.imread(input_pic)

# # image = Image.open(config.input_pic).convert('RGB')
images = transform(image).unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
    output = model(images)
    kernel_grid = vutils.make_grid(output[0], normalize=True, scale_each=True)  # 1*输入通道数, w, h
    # kernel_grid = output[0].mean(1)
    print(kernel_grid)
    kernel_grid = kernel_grid[5, :, :]
    print(kernel_grid.shape)
    print(kernel_grid)
    kernel_grid = kernel_grid * 255
    print(kernel_grid)
    norm_img = np.asarray(kernel_grid, dtype=np.uint8)
    print(norm_img)
    # 求输入图片像素最大值和最小值
    Imax = np.max(norm_img)
    Imin = np.min(norm_img)
    # 要输出的最小灰度级和最大灰度级
    Omin, Omax = 0, 255
    # 计算a 和 b的值
    a = float(Omax - Omin) / (Imax - Imin)
    b = Omin - a * Imin
    # 矩阵的线性变化
    out_image = a * norm_img + b
    # 数据类型的转化
    out_image = out_image.astype(np.uint8)


    print(out_image)
    heat_img = cv2.applyColorMap(out_image, 2) # 注意此处的三通道热力图是cv2专有的GBR排列
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)# 将BGR图像转为RGB图像


    img_add = cv2.addWeighted(img1, 0.3, heat_img, 0.7, 0)

    # vis.images(heat_img)
    # vis.images(img_add)
    cv2.namedWindow('img')
    cv2.imshow('img', img_add)
    cv2.namedWindow('heat')
    cv2.imshow('heat', heat_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# vis.heatmap(heat_img)