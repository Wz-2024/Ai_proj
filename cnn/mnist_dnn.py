import torch

from torchvision import datasets,transforms
import torch.nn as nn
import  torch.nn.functional as F
import torch.optim as optim
# 判断pytorch是否能调用GPU来运算
use_cuda=torch.cuda.is_available()
print(use_cuda)

#设置device变量
if use_cuda:
    device=torch.device('cuda')
else:
    device=torch.device('cpu')

#设置对数据进行处理的逻辑
transform=transforms.Compose([
    transforms.ToTensor(),#转化为深度学习所需的张量
    transforms.Normalize((0.1307),(0.3081))#让图片数据做标准归一化;     两个参数分别是是标准归一化的均值和方差
])
# 读取数据
datasets1=datasets.MNIST('../pytorch_study/data', train=True, download=True, transform=transform)#训练的数据集
datasets2=datasets.MNIST('../pytorch_study/data', train=False, download=True, transform=transform)#不是训练的数据集


#设置数据的加载器,,,顺便设置批次大小和是否打乱顺讯
train_loader=torch.utils.data.DataLoader(datasets1,batch_size=128,shuffle=True)
test_loader=torch.utils.data.DataLoader(datasets2,batch_size=1000)


# for batch_idx,data in enumerate(train_loader,0):
#     # inputs,targets表示x和y
#     inputs,targets=data
#     #训练集本身是 (60000,1,28,28) 60000张,单通道28*28
#     # 表示将每个28*28的图片变成784向量
#     x=inputs.view(-1,28*28)
#
#     x_std=x.std().item()#表示用张量算出标准差,, 再转化成数字
#     x_mean=x.mean().item()#算均值
#
#
# print('均值mean为'+str(x_mean))
# print('标准差std为'+str(x_std))

# 我们得出了均值是0.1306,,,标准差是0.3081

# 通过自定义的类来构建模型


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # 784表示上一层有多少个维度,,128表示当前隐藏层有多少个神经元
        #注意这里的nn.Linear只是一个∑,,,还没有引入非线性变换
        self.fc1=nn.Linear(784,128)

        # 参数表示 drop_rate
        self.dropout=nn.Dropout(0.2)
        self.fc2=nn.Linear(128,10)

    def forward(self,x):
        # 把每条数据变成一维数组
        x=torch.flatten(x,1)
        # 经过一个全连接
        # fc1表示求和
        x=self.fc1(x)
        #F.relu 表示一个非线性变换     两个函数拼接起来就表示经过了第一层全连接
        x=F.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        output=F.log_softmax(x,dim=1)
        return output

#创建一个模型实例
model=Net().to(device)


#定义训练模型的逻辑
def train_step(data,target,model,optimizer):
    optimizer.zero_grad()
    # 把数据交给模型,让它跑一个正向传播
    output=model(data)
    #nll 表示 negative log likely hood  负对数似然
    # out_put就是y预测,,,target就是真实值
    # 如果传入了一个批次的数据,,那么当前算的就是一个批次的损失
    loss=F.nll_loss(output,target)
    #进行一个反向传播,,,反向传播的本质是求梯度
    loss.backward()
    #反向传播的目的就是调参,,,接下来用optimizer来调参
    optimizer.step()
    return loss


#定义测试模型的逻辑
def test_step(data,target,model,test_loss,correct):
    output=model(data)
    #累积的批次损失
    test_loss+=F.nll_loss(output,target,reduction='sum').item()

    #后去对数概率最大值对应的索引号,这里其实就是类别号
    pred=output.argmax(dim=1,keepdims=True)
    #correct表示当前正确的数目有多少
    correct+=pred.eq(target.view_as(pred)).sum().item()
    return test_loss,correct


#创建训练调参所用的优化器
optimizer=optim.Adam(model.parameters(),lr=0.001)

#开始分轮次进行训练
EPOCHS=5
for epoch in range(EPOCHS):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target=data.to(device),target.to(device)
        loss=train_step(data,target,model,optimizer)

        if(batch_idx%10==0):
            print('Train_epoch:{}[{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(epoch,batch_idx*len(data),len(train_loader.dataset),
                                                          100.*batch_idx/len(train_loader),loss.item()))

    model.eval()
    test_loss=0
    correct=0
    # TODO: 在评估的时候不需要求梯度??
    with torch.no_grad():
        for data,target in test_loader:
            data,target=data.to(device),target.to(device)
            test_loss,correct=test_step(data,target,model,test_loss,correct)

    test_loss/=len(test_loader.dataset)
    print('\n Test set :Averge loss:{:.4f},Accuracy:{}/{} ({:.0f}%)\n'.
        format(test_loss,correct ,len(test_loader.dataset),100.*correct/len(test_loader.dataset)))
