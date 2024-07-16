import torch
import cv2
import numpy as np

def image_to_tensor(image):
    # 之前是(84, 84, 1)，下面一行就会把数据变成 (1, 84, 84)
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    return image_tensor


def resize_and_bgr2gray(image):
    # 把flappy bird每一帧图像里面的地面去除掉，小鸟每一时刻的向上或向下的选择和地面没有关系，让模型把注意力别放在地面上面
    image = image[:288, :404]  # 512*0.79=404
    # 改变大小以及变成黑白图像
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    # 二值化
    image_data[image_data > 0] = 255
    # reshape改变维度
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data