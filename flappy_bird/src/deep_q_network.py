import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

# Agent智能体的封装,
# 为了让机器能够读懂游戏画面,需要用到CNN

ACTIONS = 2


class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()
        # 输入通道是4的原因是我们会把连续4帧画面带入进来
        # 我们把每一帧其实回头会变成黑白图像
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(3136, 512)
        # 输出层
        self.fc2 = nn.Linear(512, ACTIONS)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 卷积层后面去接全连接之前要flatten一下
        x = x.view(-1, 3136)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



