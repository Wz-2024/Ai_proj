import torch
import torch.nn as nn
import os
from torchvision import datasets,transforms

from torchvision.io import read_image
from torchvision.models import AlexNet_Weights,alexnet
import cv2

#通过函数读取一张图像
image_path = r"E:\尚学堂ai\cats_and_dogs\kaggle_Dog&Cat\kaggle_Dog&Cat\validation\cats\6.jpg"
img=read_image(image_path)


# 获取模型文件
weights=AlexNet_Weights.DEFAULT
#加载网络
model=alexnet(weights=weights)
#看看模型结构
tmp=model.eval()
print(tmp)


# #测试一下模型
# #网络模型训练时候用了哪些数据转换,那测试的时候图片应该经过相同的数据转换
# preprocess=weights.transforms() #拿到对应模型函数转换的函数逻辑
# #把数据带入到数据转换的函数中
# batch=preprocess(img).unsqueeze(0)#一个批次里边装载了一张图片
#
#
# prediction=model(batch).squeeze(0).softmax(0)
# # print(prediction)
# # print(len(prediction))
# #此时,prediction里边就是一张图片对应的1000分类的概率
#
# class_id=prediction.argmax().item()
# sc=prediction[class_id].item()
# print(class_id)
# print(sc)
#
# #meta是元数据,里边包括了模型文件的一些信息
# category_name=weights.meta['categories'][class_id]
# print(f'{category_name}:{100*sc:.1f}%')
#
# # 查看下每一类是什么
# class_to_idx={cls:idx for (idx,cls) in enumerate(weights.meta['categories'])}
# print(class_to_idx)
# print(weights.meta['categories'])