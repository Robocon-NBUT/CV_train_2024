# 黄世昌的深度学习心得

## 1. 多层感知机（MLP）

多层感知机（MLP）是一种全连接的神经网络，通常由输入层、隐藏层和输出层组成。在 PyTorch 中，MLP 可以通过定义 nn.Module 子类来实现。MLP 的作用是帮助理解神经网络如何通过线性变换和非线性激活函数来学习数据的模式。它是深度学习中的基础组件之一。

## 2. 激活函数
激活函数为神经网络引入了非线性因素，使得神经网络能够学习复杂的非线性关系。在 PyTorch 中，有多种激活函数可供选择，如 ReLU、Sigmoid、Tanh 等。

## 三、3. 卷积神经网络（CNN）
卷积神经网络（CNN）是专门为处理具有网格结构数据（如图像、音频）而设计的神经网络。它通过卷积层、池化层和全连接层等组件，自动提取数据中的局部特征，从而在图像识别、目标检测等领域取得了巨大的成功。
CNN 实现
python
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 56 * 56, 128)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = x.view(-1, 16 * 56 * 56)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x


## 4. 池化层

池化层是 CNN 中的重要组成部分，主要包括最大池化（Max Pooling）和平均池化（Average Pooling）。
最大池化
最大池化在给定的池化窗口内选择最大值作为输出。例如，在一个 2x2 的池化窗口中，它会从这 4 个值中选取最大的一个作为输出值。在 CNN 实现中，我们使用 nn.MaxPool2d 来实现最大池化。
python
self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
平均池化
平均池化则是计算池化窗口内所有值的平均值作为输出。虽然平均池化在保留空间信息方面有一定优势，但在实际应用中，最大池化因其能够更好地保留重要特征而更为常用。

## 5. 损失函数

损失函数用于衡量模型预测结果与真实标签之间的差异，它是模型训练过程中的重要指导。在 PyTorch 中，有多种损失函数可供选择，具体取决于任务的类型。
常见损失函数
交叉熵损失（Cross Entropy Loss）：常用于分类任务。它结合了 Softmax 函数和负对数似然损失，能够有效地衡量模型预测的概率分布与真实标签之间的差异。在多分类问题中，我们通常使用 nn.CrossEntropyLoss。
python
criterion = nn.CrossEntropyLoss()
均方误差损失（Mean Squared Error Loss，MSE Loss）：主要用于回归任务，它计算预测值与真实值之间差值的平方的平均值。在 PyTorch 中，通过 nn.MSELoss 实现。

## 6. 优化器
优化器负责在模型训练过程中根据损失函数的梯度来更新模型的参数，以达到最小化损失函数的目的。PyTorch 提供了多种优化器，如随机梯度下降（SGD）、Adagrad、Adadelta、Adam 等。
Adam 优化器
Adam 优化器是一种常用的优化器，它结合了 Adagrad 和 RMSProp 的优点，能够自适应地调整学习率。在实践中，Adam 优化器通常能够在各种任务中取得较好的效果。
python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
