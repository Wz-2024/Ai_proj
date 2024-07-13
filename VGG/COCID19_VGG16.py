import os
import time

import torch
import  torchvision
from torchvision.models import vgg16,VGG16_Weights
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader



#检测cuda
device=torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

#构建预训练模型
model=torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
#将这个预训练模型的所有参数都固定(就是不用再求梯度更新优化他们
for param in model.parameters():
    param.requires_grad=False

#现在重写全连接层,,这样这部分的 param.requires_grad的值为True
model.classifier=nn.Sequential(
    nn.Linear(512*7*7,256),
    #512表示最后一层卷积得到的特征图层数 7*7表示池化后的大小  256是当前这一层神经元的个数
    nn.ReLU(inplace=True),
    nn.Linear(256,2)#这个256是和上边的256相对应的,,2表示现在要做二分类
)

#接下来的写法是不去训练所有的参数


#首先查看下那些参数在传播的过程中会被训练
print('当前需要参与训练的参数(层)为:')
params_to_update=[]
for name,param in model.named_parameters():#这个函数可以找出哪些参数是需要被训练的
    if param.requires_grad:
        params_to_update.append(param)
        print("\t",name)
'''
这里输出的0和2就是对应的全连接 weight和bias表示权重参数和偏移量参数
'''
model=model.to(device)
#定义transform
data_transform={
    'train':transforms.Compose([
        transforms.Resize((150,150)),
        transforms.RandomRotation(40),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'test':transforms.Compose([
        transforms.Resize((150,150)),
        transforms.ToTensor(),
    ])
}

#读取数据
base_dir=r'E:\尚学堂ai\cats_and_dogs\kaggle_Dog&Cat\kaggle_Dog&Cat'
train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')

train_datasets=datasets.ImageFolder(train_dir,transform=data_transform['train'])
test_datasets=datasets.ImageFolder(validation_dir,transform=data_transform['test'])

#这里其实就是把train中的两个文件夹拿出来
example_class=train_datasets.classes
index_classes=train_datasets.class_to_idx
print(example_class)
print(index_classes)

train_dataloader=DataLoader(dataset=train_datasets,batch_size=10,shuffle=True)
test_dataloader=DataLoader(dataset=test_datasets,batch_size=10)

loss_f=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.RMSprop(model.parameters(),lr=0.001)

EPOCHS=20
for epoch in range(EPOCHS):
    print(epoch)

    model.train()
    running_loss=0.0
    running_corrects=0

    start_time=time.time()

#train_dataloader的参数是datasets,,datasets是两个文件夹,猫和狗
    for bat_idx,(data,target) in enumerate(train_dataloader):
        X,y=data.to(device),target.to(device)
        optimizer.zero_grad()
        y_pred=model(X)
        loss=loss_f(y_pred,y)
        loss.backward()
        optimizer.step()

        #统计
        running_loss+=loss.item()
        pred=y_pred.argmax(dim=1,keepdim=True)
        running_corrects+=pred.eq(y.view_as(pred)).sum().item()


    epoch_loss=running_loss*10/len(train_datasets)
    epoch_acc=running_corrects*100/len(train_datasets)
    print(f"Train Loss:{epoch_loss:.4f},Acc:{epoch_acc:.2f}")


    #计算一轮计算花费的时长
    end_time=time.time()
    time_taken=end_time-start_time
    print(f"time:{time_taken}")

    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            X, y = data.to(device), target.to(device)

            y_pred = model(X)  # 调用模型
            # 顺带算一下测试集上的损失(正向传播,因为是测试模式,所以是没有反向传播的)
            loss = loss_f(y_pred, y)
            test_loss += loss.item()
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    epoch_loss = test_loss * 20 / len(test_datasets)
    epoch_acc = correct * 100 / len(test_datasets)
    print(f"Test Loss:{epoch_loss:.4f},Test Acc:{epoch_acc:.2f}%")
