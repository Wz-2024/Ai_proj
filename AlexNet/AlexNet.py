import torch
import torch.nn as nn
import os
from torchvision import datasets, transforms
from torchvision.io import read_image
import torch.utils.data
from torchvision.models import alexnet, AlexNet_Weights
import time

# image_path = r"E:\尚学堂ai\cats_and_dogs\kaggle_Dog&Cat\kaggle_Dog&Cat\validation\cats\6.jpg"

# 检测cuda是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

weights = AlexNet_Weights.DEFAULT
model = alexnet(weights=weights)

# 选择性的调整参数:比如现在不调整特征提取的层(卷积层)的参数,,只调整全连接层(classifier)的参数
for k, v in model.named_parameters():
    if not k.startswith('classsifier'):
        # 除了classifier以外的层,都不调整参数
        v.requires_grad = False

# 替换预训练模型中的最后的一些全连接层和输出层
model.classifier[1] = nn.Linear(9216, 4096)
model.classifier[4] = nn.Linear(4096, 4096)
# 做二分类
model.classifier[6] = nn.Linear(4096, 2)

model = model.to(device)

transforms_for_train = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomRotatio(40),
    # transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transforms_for_test = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomRotatio(40),
    # transforms.ColorJitter(),
    transforms.ToTensor(),
])

base_dir = 'E:/尚学堂ai/cats_and_dogs/kaggle_Dog&Cat/kaggle_Dog&Cat'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_datasets = datasets.ImageFolder(train_dir, transform=transforms_for_train)
test_datasets = datasets.ImageFolder(validation_dir, transform=transforms_for_test)

train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=20, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=20)

loss_f = torch.nn.CrossEntropyLoss()  # 二分类交叉熵
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)  # 这个是优化器

EPOCHS = 100
for epoch in range(EPOCHS):
    print(epoch)

    model.train()
    running_loss = 0.0
    running_corrects = 0

    start_time = time.time()
    # print(23)
    for batch_idx, (data, target) in enumerate(train_dataloader):
        X, y = data.to(device), target.to(device)
        # print(23)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_f(y_pred, y)
        loss.backward()
        optimizer.step()
        # 这里的optimizer.step()通常要跟早backward()的后边,，
        # 因为 loss.backward() 会计算损失函数相对于模型参数的梯度，
        # 而 optimizer.step() 则使用这些梯度来更新参数。
        running_loss += loss.item()
        # keepdims表示传进来的y_pred是几个维度的,那么当前得到的pred仍然是几个维度
        pred = y_pred.argmax(dim=1, keepdims=True)
        # 计算结果中正确的数量,,方法是用y_pred和pred进行比较
        running_corrects += pred.eq(y.view_as(pred)).sum().item()

    epoch_loss = running_loss * 20 / len(train_datasets)
    epoch_acc = running_corrects * 100 / len(train_datasets)
    print("Train loss:{:.4f} Acc:{:.2f}".format(epoch_loss, epoch_acc))

    # 计算一轮所需的时间
    end_time = time.time()
    time_taken = end_time - start_time
    print("time:", time_taken, sep="  ")

    # 进入评估模式
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            X, y = data.to(device), target.to(device)

            y_pred = model(X)#调用模型
            # 顺带算一下测试集上的损失(正向传播,因为是测试模式,所以是没有反向传播的)
            loss = loss_f(y_pred, y)
            test_loss+=loss.item()
            pred=y_pred.argmax(dim=1,keepdim=True)
            correct+=pred.eq(y.view_as(pred)).sum().item()

    epoch_loss=test_loss*20/len(test_datasets)
    epoch_acc=correct*100/len(test_datasets)
    print(f"Test Loss:{epoch_loss:.4f},Test Acc:{epoch_acc:.2f}%")
