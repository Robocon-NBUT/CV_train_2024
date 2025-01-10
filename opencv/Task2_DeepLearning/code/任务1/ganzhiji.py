'''
import torch

from torch import nn
from d2l import torch as d2l



def relu(x):
    a=torch.zeros_like(x)
    return torch.max(x,a)

def net(x):
    x=x.reshape((-1,num_in))
    h=relu(x@w1 + b1)
    return h@w2 + b2



batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_in ,num_out ,num_hid =784,10,256

w1=nn.Parameter(torch.randn(num_in,num_hid,requires_grad=True))

b1=nn.Parameter(torch.zeros(num_hid,requires_grad=True))

w2=nn.Parameter(torch.randn(num_hid,num_out,requires_grad=True))

b2=nn.Parameter(torch.zeros(num_out,requires_grad=True))

params=[w1,b1,w2,b2]




loss = nn.CrossEntropyLoss()

num_ep,lr =10,0.1
updater = torch.optim.SGD(params,lr=lr)
print(d2l.train_ch3(net,train_iter,test_iter,loss,num_ep,updater))

'''

'''
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import numpy as np


# 定义ReLU激活函数，保持不变
def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x, a)


# 定义神经网络的前向传播函数net，保持不变，但要确保num_in在使用前已正确定义（下面会处理）
def net(x):
    x = x.reshape((-1, num_in))
    h = relu(x @ w1 + b1)
    return h @ w2 + b2


# 定义输入特征数量、输出类别数量、隐藏层神经元数量，这里直接明确赋值，保持与原代码一致
num_in, num_out, num_hid = 784, 10, 256

# 随机初始化从输入层到隐藏层的权重w1，保持不变
w1 = nn.Parameter(torch.randn(num_in, num_hid, requires_grad=True))
# 初始化隐藏层的偏置b1为全零张量，保持不变
b1 = nn.Parameter(torch.zeros(num_hid, requires_grad=True))
# 随机初始化从隐藏层到输出层的权重w2，保持不变
w2 = nn.Parameter(torch.randn(num_hid, num_out, requires_grad=True))
# 初始化输出层的偏置b2为全零张量，保持不变
b2 = nn.Parameter(torch.zeros(num_out, requires_grad=True))

# 将所有需要更新的网络参数（权重和偏置）收集到一个列表中，保持不变
params = [w1, b1, w2, b2]

# 选用交叉熵损失函数（常用于多分类问题），用于衡量模型预测输出与真实标签之间的差异，保持不变
loss = nn.CrossEntropyLoss()

# 定义训练轮数为10，学习率为0.1，保持不变
num_ep, lr = 10, 0.1
# 使用随机梯度下降优化器（SGD），传入网络参数列表params和学习率lr，用于在训练过程中更新网络参数，使损失函数尽可能减小，保持不变
updater = torch.optim.SGD(params, lr=lr)

# 数据加载部分，使用torchvision加载时尚MNIST数据集，以下是具体步骤
# 定义数据预处理操作，将图像转换为张量，并进行归一化（这里的归一化参数可根据实际情况调整）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载训练集，设置批量大小为256，并应用定义好的预处理操作，同时打乱数据顺序
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                              download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True)

# 加载测试集，同样设置批量大小为256并应用预处理操作
testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                             download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                         shuffle=False)

# 训练相关函数实现，替代原代码中的d2l.train_ch3函数
def train(model, trainloader, testloader, loss_fn, num_epochs, optimizer):
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_correct = 0
        model.train()
        for data, target in trainloader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_correct += (predicted == target).sum().item()

        train_loss /= len(trainloader)
        train_accuracy = 100. * train_correct / len(trainloader.dataset)

        test_loss = 0.0
        test_correct = 0
        model.eval()
        with torch.no_grad():
            for data, target in testloader:
                output = model(data)
                loss = loss_fn(output, target)
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                test_correct += (predicted == target).sum().item()

        test_loss /= len(testloader)
        test_accuracy = 100. * test_correct / len(testloader.dataset)

        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    return train_accuracy, test_accuracy


# 创建模型实例（确保num_in已正确定义，这里是为了符合Python语法要求提前定义）
num_in = 784
model = net
# 调用自定义的训练函数进行训练，并打印训练结果（这里传递的model实际上是函数net，也可以将net封装为类的形式然后传递实例化对象）
train_accuracy, test_accuracy = train(model, trainloader, testloader, loss, num_ep, updater)
print(f"Final Train Accuracy: {train_accuracy:.4f}, Final Test Accuracy: {test_accuracy:.4f}")

'''

import torch
import torch.nn as nn
import torch.optim as optim


# 定义多层感知机模型
class XORMLP(nn.Module):
    def __init__(self):
        super(XORMLP, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out



# 训练数据
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 初始化模型、损失函数和优化器
model = XORMLP()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 训练模型
for epoch in range(10000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/10000], Loss: {loss.item():.4f}')


input_str = input("请输入两个0或1，以空格分隔: ")
input_list = list(map(int, input_str.split()))
input_tensor = torch.tensor(input_list, dtype=torch.float32).unsqueeze(0)


# 测试模型
with torch.no_grad():
    test_outputs = model(X)
    test_output = model(input_tensor)
    exp_predicted = (test_outputs >= 0.5).float()
    predicted = (test_output >= 0.5).float()
#    print("预测结果:")
#    print(exp_predicted)
    print("实际结果:")
    print(predicted)
