import torch.nn as nn
import torch.nn.functional as f
# Agent智能体的封装,
# 为了让机器能够读懂游戏画面,需要用到CNN

ACTION = 2


class DQNNetwork(nn.Module):
    def __init__(self):
        super(DQNNetwork).__init__()
        #输入通道为4,原因是将彩色图像变为灰度图(一个通道),然后将连续4帧画面带入进来,,
        self.conv1=nn.Conv2d(in_channels=4,out_channels=32,kernel_size=8,stride=4,padding=0)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2,padding=0)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=0)
        self.fc1=nn.Linear(1600,512)
        self.fc1=nn.Linear(512,ACTION)




        def forward(self,x):
            f.relu(self.conv1(x))
            f.relu(self.conv2(x))
            f.relu(self.conv3(x))
            #卷积层后面接全连接层前需要flatten一下
            x=x.view(-1,1600)
            x=f.relu(self.fc(1))
            x=self.fc2(x)
            return x



