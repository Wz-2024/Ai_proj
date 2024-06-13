import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
import torch.utils.data
import matplotlib.pyplot as plt
# 检测CUDA是否可用
use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# 构建网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        # Linear中的第一个数值现在不知道该是多少，就等报错之后再修改
        self.fc1 = nn.Linear(18496, 512)
        # 下面的2指的是输出层有两个节点，做二分类
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # 最后返回的是对数几率，其实如果是预测，我们将对数几率去最大，对应的索引号就是预测的类标号
        # 如果是训练的时候，那就看后面的损失函数loss function要的是不是对数几率
        return x


transforms_for_train = transforms.Compose([
    # 改变图片的大小
    transforms.Resize((150, 150)),

    # 下面是关于使用数据增强的部分
    transforms.RandomRotation(40),  # 可用随机旋转角度
    transforms.RandomHorizontalFlip(),  # 水平翻转
    # transforms.RandomVerticalFlip(),  # 上下翻转
    # transforms.RandomCrop(150),  # 从原图扣指定大小的区域，扣一个方块区域
    # transforms.RandomResizedCrop(150),  # 随机的扣取不同高宽比的图，缩放成指定大小
    # transforms.ColorJitter(),  # 随机的改变图片的亮度、对比度、饱和度、色调
    # transforms.RandomAffine(degrees=40, translate=(0.2, 0.2), scale=(0.2, 0.2), shear=0.2),  # 仿射变换

    # 把图片数据的每个数值缩放到0到1之间，并且图片类型变成Tensor张量
    transforms.ToTensor()
])

transforms_for_test = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])


if __name__ == '__main__':
    # 创建一个模型实例
    model = Net().to(device)

    base_dir =r'E:\尚学堂ai\cats_and_dogs\kaggle_Dog&Cat\kaggle_Dog&Cat'

    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    train_datasets = datasets.ImageFolder(train_dir, transform=transforms_for_train)
    test_datasets = datasets.ImageFolder(validation_dir, transform=transforms_for_test)

    example_classes = train_datasets.classes
    print(example_classes)
    index_classes = train_datasets.class_to_idx
    print(index_classes)

    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=20, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=20, num_workers=4)

    # 开始训练前，首先设置损失函数
    # 之所以我们前面Net最后输出的是没有归一化的对数几率，就是因为此处的交叉熵损失需要的input是未归一化的对数几率
    # 但是又因为交叉熵损失是针对Softmax设计的，所以哪怕是做二分类，我们也得输出层是2个神经元
    loss_f = torch.nn.CrossEntropyLoss()
    # 总结一下，在PyTorch中进行二分类，有三种主要的全连接层，激活函数和loss_function组合的方法，分别是：
    # torch.nn.Linear+torch.sigmoid + torch.nn.BCELoss，
    # torch.nn.Linear+     BCEWithLogitsLoss，和
    # torch.nn.Linear（输出维度为2）+     torch.nn.CrossEntropyLoss，
    # 后两个loss function分别集成了   Sigmoid(二分类) 和Softmax(多分类)。

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

    # 开始分轮次分批次训练
    train_acc_list = []
    test_acc_list = []


    EPOCHS = 500
    for epoch in range(EPOCHS):
        print(epoch)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        # batch_size=20,因此这里会循环执行625次
        for batch_idx, (data, target) in enumerate(train_loader):
            # 获取一个批次的数据图片X和标签y
            X, y = data.to(device), target.to(device)
            # 开始每一次正向反向传播之前，把optimizer重置一下
            optimizer.zero_grad()
            # 正向传播从 X->predictions
            y_pred = model(X)
            # 从 predictions -> Loss值, loss计算的是一个批次所有样本的平均损失
            loss = loss_f(y_pred, y)
            # 反向传播,求导
            loss.backward()
            # 应用求出来的梯度更新参数
            optimizer.step()

            # .item()相当于是获取tensor的数值
            running_loss += loss.item()
            # 获取预测的类别号
            pred = y_pred.argmax(dim=1, keepdims=True)
            # 计算一个批次模型预测对了多少个
            running_corrects += pred.eq(y.view_as(pred)).sum().item()

        # 打印，20对应的batch_size，running_loss * 20 得到的就是一个轮次所有样本的损失
        # len(train_datasets) 对应的就是训练集整体的样本数
        epoch_loss = running_loss * 20 / len(train_datasets)  # 得到的是平均损失
        epoch_acc = running_corrects * 100 / len(train_datasets)
        train_acc_list.append(epoch_acc)
        print("Training Loss:{:.4f} Acc:{:.4f}%".format(epoch_loss, epoch_acc))

        model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                X, y = data.to(device), target.to(device)

                y_pred = model(X)
                loss = loss_f(y_pred, y)
                test_loss += loss.item()
                pred = y_pred.argmax(dim=1, keepdims=True)
                correct += pred.eq(y.view_as(pred)).sum().item()

        epoch_loss = test_loss * 20 / len(test_datasets)
        epoch_acc = correct * 100 / len(test_datasets)
        test_acc_list.append(epoch_acc)
        print("Test Loss:{:.4f} Acc:{:.4f}%".format(epoch_loss, epoch_acc))

 # 绘制训练和测试准确率图
    plt.figure(figsize=(10, 5))
    plt.plot(range(EPOCHS), train_acc_list, label='Training Accuracy')
    plt.plot(range(EPOCHS), test_acc_list, label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.show()
