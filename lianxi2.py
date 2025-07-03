import cv2
import numpy as np
from PIL import Image


input_pic = 'data/test/images/303.png'
img1 = cv2.imread(input_pic)
img = cv2.imread(input_pic, 2)
print(img)
norm_img = np.zeros(img.shape)
cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
norm_img = np.asarray(img, dtype=np.uint8)

heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET) # 注意此处的三通道热力图是cv2专有的GBR排列
heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)# 将BGR图像转为RGB图像


img_add = cv2.addWeighted(img1, 0.3, heat_img, 0.7, 0)


cv2.namedWindow('img')
cv2.imshow('img', img_add)
cv2.namedWindow('heat')
cv2.imshow('heat', heat_img)
cv2.waitKey(0)
cv2.destroyAllWindows()