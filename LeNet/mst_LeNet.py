import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms,datasets
from torch.utils.data import DataLoader

# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet,self).__init__()
#         self.conv1=nn.Conv2d(1,6,5,1)
#         self.conv2=nn.Conv2d(6,16,5,1)
#         self.conv3=nn.Conv2d(16,120,4,1)
#         self.fc1=(120,64)
#         self.fc2=(64,10)
#
#     def forward(self,x):
#         x=self.conv1(x)
#         x=F.sigmoid(x)
#         x=F.max_pool2d(x,2)
#
#         x=self.conv2(x)
#         x=F.sigmoid(x)
#         x=F.max_pool2d(x,2)
#
#         x=self.conv3(x)
#
#         #reshape  要把四维的转化为一维的
#         x=x.view(x.shape[0],-1)
#         x=self.fc1(x)
#         x=F.sigmoid(x)
#         x=self.fc2(x)
#
#         return x
#
# net = LeNet()


net=nn.Sequential(
    nn.Conv2d(1,6,5,padding=2),nn.Sigmoid(),#卷积
    nn.AvgPool2d(kernel_size=2,stride=2),#池化
    nn.Conv2d(6,16,5),nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    #注意这里是两层卷积+三层全连接,,本质上是一样的,,把之前的卷积层用全连接层替代
    nn.Flatten(),
    #这里代替了第三层卷积
    nn.Linear(16*5*5,120),nn.Sigmoid(),
    nn.Linear(120,64),nn.Sigmoid(),
    nn.Linear(64,10)
)
#在pytorch中,默认的图片形状是(m,c,h,w)
# 分别对应 batch_size,channel,height,width


# 打印每组对应的数据
X=torch.rand(size=(1,1,28,28),dtype=torch.float32)

for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape :\t',X.shape)


transform=transforms.Compose([
    #转化为pytorch的Tensor,把Tensor的数值缩放到0~1之间
    transforms.ToTensor(),
    #0.1307和0.3018是根据数据集得到的均值和方差
    transforms.Normalize((0.1307),(0.3081,))

])

train_dataset=datasets.MNIST(root='./LeNet',train=True,transform=transform,download=True)
test_dataset=datasets.MNIST(root='./LeNet',train=True,transform=transform,download=True)


train_loader=DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(dataset=train_dataset,batch_size=64,shuffle=False)

#交叉熵
loss_fn=torch.nn.CrossEntropyLoss()

use_gpu=torch.cuda.is_available()
if use_gpu:
    net.cuda()
    loss_fn.cuda()


#Adam是优化器
#它结合了AdaGrad和RMSProp的优点，通过计算梯度的一阶和二阶矩估计来调整学习率。
#Adam优化器通常比简单的随机梯度下降（SGD）收敛更快，并且在处理稀疏梯度和非平稳目标时表现良好。
optimizer=torch.optim.Adam(net.parameters(),lr=0.001)

def train(epoch_id):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader):
        inputs,target=data
        if use_gpu:
            inputs,target=inputs.cuda(),target.cuda()

        optimizer.zero_grad()
        # x->y_hat
        outputs = net(inputs)
        #对比y_hat和target(真实y) 得到loss
        loss=loss_fn(outputs,target)

        #loss->grads
        loss.backward()
        #grads->调参
        optimizer.step()

        # 把Tensor的值取出来
        running_loss+=loss.item()
        #每隔300轮打印一次
        if batch_idx%300==0:
            print('[%d , %5d] loss:%.3f'%(epoch_id+1,batch_idx+1,running_loss))
            running_loss=0.0#每隔300置零一次

def test():
    correct=0
    total=0
    with torch.no_grad():#在测试过程中,不需要梯度计算
        for data in test_loader:
            images,labels=data
            if use_gpu:
                images,labels=images.cuda(),labels.cuda()

            outputs=net(images)

            # outputs.data：从模型的输出张量中获取数据部分，去掉了与计算图相关的信息（如梯度）。
            # torch.max(outputs.data, dim=1)：计算在给定维度上（在这里是dim = 1，即每一行）
            # 最大值。这个函数返回两个值：每行的最大值和最大值所在的索引。
            # _, predict：下划线
            # _表示忽略第一个返回值（即每行的最大值），只保留第二个返回值
            # predict（即最大值所在的索引），这些索引对应于预测的类别标签。

            _,predict=torch.max(outputs.data,dim=1)
            total+=labels.size(0)
            #得到的结果符合预测值就是正确结果
            correct+=(predict==labels).sum().item()
    print("Accuracy on the test set %.2f %%" % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(5):
        train(epoch_id=epoch)
        if(epoch%2==0):
            test()
