

image_path = r"E:\尚学堂ai\cats_and_dogs\kaggle_Dog&Cat\kaggle_Dog&Cat\validation\cats\6.jpg"
import torch
import torch.nn as nn
import os
from torchvision import datasets,transforms
from torchvision.io import read_image
from torchvision.models import alexnet,AlexNet_Weights

img =read_image(r"E:\尚学堂ai\cats_and_dogs\kaggle_Dog&Cat\kaggle_Dog&Cat\validation\cats\15.jpg")
#获取模型文件
weights=AlexNet_Weights.DEFAULT
#让模型加载参数
model=alexnet(weights=weights)
#模型结构

tmp=model.eval()
print(tmp)
# 从输出结果来看,,model已经是写好的,比较完善的模型,包括了参数的设置,因此不需要再手写一遍


#测试模型
#Tip:网络模型在训练时经过了哪些(增强处理),那么在测试时,应该对数据集也做出相应的处理
#preprecess相当于拿到了当时训练时,用到的数据转化(增强处理)
preprocess=weights.transforms()#拿到了对应模型的数据转化的函数逻辑

#将数据带入到数据转化的函数中
batch=preprocess(img).unsqueeze(0)
prediction=model(batch).squeeze(0).softmax(0)#正向传播  inference   X->y_pred

'''
print(prediction)  打印出当前的概率分布,,它是一个列表,每个元素是每个分类对应的概率
print(len(prediction)) 长度为1000,表示当前共1000个分类
'''
mx_class_id=prediction.argmax().item()
print("当前概率最高的索引是"+str(mx_class_id))
print("对应的概率为"+str(prediction[mx_class_id].item()))
#meta是元数据,里边包括了模型文件的一些信息
class_name=weights.meta['categories'][mx_class_id]
print(f"{class_name}:{100*prediction[mx_class_id].item()}%")


#接下来看一下这1000个分类,,每个索引号对应的是什么
class_id={cls:idx for (idx,cls) in enumerate(weights.meta['categories'])}

print(class_id)

