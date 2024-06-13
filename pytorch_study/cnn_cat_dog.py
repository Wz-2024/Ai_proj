import os

import torch
import torch.nn as nn
import torch.nn.functional as F
# datasets就是读数据集用的,transform是做转换
from torchvision import datasets,transforms
import torch.utils.data


use_cuda=torch.cuda.is_available()
if use_cuda:
    device=torch.device('cuda')
else:
    device=torch.device('cpu')

#构建网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # 输入通道数,卷积核个数,卷积核大小,步长
        self.conv1=nn.Conv2d(3,16,3,1)
        self.conv2=nn.Conv2d(16,32,3,1)
        self.conv3=nn.Conv2d(32,64,3,1)

        # 对于这个全连接层,,输出数就是隐藏结点,可以自定义地写,,但是输入层和上一次输出有关,
        # 报错的时候该就行了
        self.fc1=nn.Linear(18496,512)

        # 由于上一层的输出就是512,所以这里的输入肯定是512,,,输出为2表示还需要softmax-多分类交叉熵,,
        #1就是sigmoid 二分类交叉熵    输出层有两个神经元做二分类
        self.fc2=nn.Linear(512,2)

    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2)

        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x=torch.flatten(x,1)
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)

        #最后返回对数几率,如果是预测,,那就找出几率最大的对应的索引号
        #如果是训练的时候,那就看后边的损失函数loss function要的是不是对数几率
        return x

if __name__=='__main':
    #创建一个模型实例
    model=Net().to(device)
    transforms_for_train= transforms.Compose([
        #改变图片大小
        transforms.Resize((150,150)),
        #数据增强    下边的增强一般就写1~2个
        transforms.RandomRotation(40),#随机旋转角度
        transforms.RandomHorizontalFlip(),#翻转
        transforms.RandomVerticalFlip(),
        #这两个扣图的其实并不常用
        # transforms.RandomCrop(150),#随机地扣出一些区域
        # transforms.RandomResizedCrop(150),#随机地扣出一些区域,,然后又缩放至150*150
        #仿射变换
        transforms.RandomAffine(degrees=40,translate=(0.2,0.2),scale=(0.2,0.2),shear=0.2),
        #随机改变亮度,对比度,饱和度,色调,,,其实就是改变图片的光学属性
        transforms.ColorJitter(),



        #将图片的每个数值缩放到0-1之间,并且将图片转换为Tensor张量
        transforms.ToTensor()
    ])
    transforms_for_test=transforms.Compose([
        transforms.Resize((150,150)),
        transforms.ToTensor()
    ])

    base_dir=r'E:\尚学堂ai\cats_and_dogs\kaggle_Dog&Cat\kaggle_Dog&Cat'
    train_dir=os.path.join(base_dir,'train')
    validation_dir=os.path.join(base_dir,'test')

    train_datasets=datasets.ImageFolder(train_dir,transform=transforms_for_train)
    test_datasets=datasets.ImageFolder(validation_dir,transform=transforms_for_test)

    example_class=train_datasets.classes
    print(example_class)
    index_class=train_datasets.class_to_idx
    print(index_class)

    train_loader=torch.utils.data.DataLoader(train_datasets,batch_size=20,shuffle=True,num_workers=4)
    test_loader=torch.utils.data.DataLoader(train_datasets,batch_size=20,num_workers=4)


    #开始训练
    #前边Net最后的输出是没有归一化的对数几率,,,是因为此处的交叉熵损失需要的input就是未归一化的对数几率
    #由于交叉熵损失是针对softmax设计的,,,所以即便做2分类,,,输出层也需要两个神经元
    loss_f=torch.nn.CrossEntropyLoss()
    # pytorch中做二分类共有三种 全连接层,激活函数和loss 的组合方式
    #1.  torch.nn.Linear +torch.sigmoid+torch.nn.BCELoss
    #2.  torch.nn.Linear+BCEWithLogitsLoss (第二个集成了sigmoid)
    #3.  torch.nn.Linear(输出维度为2)+torch.nn.CrossEntropyLoss(第二个集成了softmax)

    optimizer=torch.optim.RMSprop(model.parameters(),lr=0.001)


    #开始分轮次分批次训练
    EPOCH=100
    for epoch in range(EPOCH):
        print(epoch)
        model.train()
        running_loss=0.0
        running_corrects=0
        for batch_idx, (data,target) in enumerate(train_loader):
            #获取一个批次的数据图片X和标签y
            X,y=data.to(device),target.to(device)
            #开始每一次正向传播之前,把optimizer重置一下
            optimizer.zero_grad()
            #正向传播X->predictions
            y_pred=model(X)
            #从predictions->loss值   计算的是一个批次所有样本的平均损失
            loss=loss_f(y_pred,y)
            #反向传播,,求导
            loss.backward()
            #用求出来的梯度更新参数
            optimizer.step()
            #.item()相当于获取Tensor的数值
            running_loss+=loss.item()
            #获取预测的类别号
            pred=y_pred.argmax(dim=1,keepdims=True)
            #计算一个批次模型预测对了多少个
            running_corrects+=pred.eq(y.view_as(pred)).sum().item()

        #这里打印损失,,,20对应batch_size,running_loss*20得到的是一个轮次整体的损失
        #len(train_datasets) 表示训练集整体的样本数
        epoch_loss=running_loss*20/len( )  #平均损失
        epoch_acc=running_corrects*100/len(train_datasets)
        print("training loss:{:.4f} Acc:{:.4f}%".format(epoch_loss,epoch_acc))

        #验证模式
        model.eval()
        test_loss=0.0
        correct=0
        with torch.no_grad():
            for data,target in test_loader:
                X,y=data.to(device),target.to(device)
                y_pred=model(X)
                loss=loss_f(y_pred,y)
                test_loss+=loss.item()
                pred=y_pred.argmax(dim=1,keepdims=True)
                correct+=pred.eq(y.view_as(pred)).sum().item()

        epoch_loss=test_loss*20/len(test_datasets)
        epoch_acc=correct*100/len(test_datasets)
        print("Test Loss:{:.4f} Acc:{:.4f}".format(epoch_loss,epoch_acc))