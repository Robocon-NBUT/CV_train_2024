# Task 1 Deep Learning
## 学习内容
 - python 版 pytorch库的安装与使用 
 - 安装：
   可以使用 pip 命令进行安装，例如 pip install torch torchvision。对于不同的操作系统和硬件（如 GPU 支持），可能需要额外的步骤，例如使用 CUDA 支持 GPU 加速时，需要确保 CUDA 工具包的安装并安装与 CUDA 版本匹配的 PyTorch 版本。
   还可以使用 Anaconda 环境管理工具，通过 conda 命令进行安装，这样可以更方便地管理不同项目的依赖和环境。
   使用：
   PyTorch 提供了张量（Tensor）作为核心数据结构，类似于 NumPy 的数组，但支持 GPU 加速，可通过 torch.Tensor 进行创建和操作。
   支持自动微分功能，这是深度学习中实现反向传播的关键，通过 requires_grad=True 可以追踪张量的梯度，使用 backward() 方法计算梯度，这在优化神经网络参数时至关重要。
   提供了大量的工具和函数，如 torch.nn 模块用于构建神经网络，torch.optim 用于优化器，torch.utils.data 用于数据加载和预处理等。

 - 创建神经网络模型类
 - 继承 nn.Module：
   在 PyTorch 中，通常创建一个继承自 nn.Module 的类来定义神经网络。在 __init__ 方法中定义网络的层，如 nn.Linear 用于全连接层，nn.Conv2d 用于卷积层等。
   实现 forward 方法来定义数据在网络中的前向传播逻辑，该方法定义了输入数据如何通过网络的各个层并最终得到输出。

 - 多层感知机
 - 结构：
   由输入层、一个或多个隐藏层和输出层组成，层与层之间全连接。每个神经元接收上一层的输出作为输入，并通过线性变换和激活函数计算自身的输出。
   例如：
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # 输入层到第一个隐藏层，输入维度为 2，输出维度为 4
        self.fc2 = nn.Linear(4, 1)  # 第一个隐藏层到输出层，输入维度为 4，输出维度为 1

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # 使用 sigmoid 作为激活函数
        x = torch.sigmoid(self.fc2(x))  # 使用 sigmoid 作为激活函数
        return x
   训练过程：
   包括前向传播、计算损失、反向传播和更新参数。通过 loss.backward() 计算梯度，然后使用优化器（如 optim.SGD）的 step() 方法更新参数。

 - 激活函数
 - 作用：
   引入非线性因素，使神经网络能够学习复杂的非线性关系。
   常见的激活函数包括 sigmoid 函数（将输出映射到 0 到 1 之间）、tanh 函数（将输出映射到 -1 到 1 之间）、ReLU 及其变种（如 LeakyReLU），其中 ReLU 是最常用的，它将负数置为 0，正数保持不变，能缓解梯度消失问题。

 - 卷积神经网络
 - 结构：
   包含卷积层（nn.Conv2d）、池化层（nn.MaxPool2d 或 nn.AvgPool2d）和全连接层。
   卷积层通过卷积核在输入上滑动进行卷积操作，提取特征。
   池化层用于减少特征图的尺寸，保留主要特征，降低计算量，如最大池化取区域内的最大值，平均池化取区域内的平均值。
   例如：
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # 输入通道为 1，输出通道为 32，卷积核大小为 3x3
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化层，核大小为 2x2，步长为 2
        self.fc1 = nn.Linear(32 * 13 * 13, 10)  # 全连接层，根据输入尺寸调整输入维度

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 13 * 13)  # 展开张量
        x = self.fc1(x)
        return x
   优势：
   自动学习数据的空间特征，对于图像、音频等数据非常有效，因为它们具有局部相关性。

 - 池化层 
 - 最大池化：
   如 nn.MaxPool2d，在输入特征图的每个局部区域内选取最大值作为输出，保留了最显著的特征信息，可增强对平移和旋转的不变性。
   有助于减少特征图的尺寸，降低计算量和参数数量，避免过拟合。

 - 损失函数
 - 均方误差（MSE）：
   常用于回归任务，如 nn.MSELoss，计算预测值和真实值之间的平方误差的平均值，适合输出为连续值的任务。

 - 优化器
 - 随机梯度下降（SGD）：
   如 optim.SGD，根据梯度下降的原理更新参数

 - yolov11的训练与使用
 - YOLO 系列简介：
   YOLO（You Only Look Once）是一种目标检测算法，将目标检测视为回归问题，将图像划分为网格，每个网格预测边界框和类别概率。
   YOLOv11 可能是一个假设的 YOLO 版本，一般来说，YOLO 系列算法在速度和精度上有较好的平衡。
   训练过程：
   准备数据集，标注包含物体的类别和边界框信息。
   构建 YOLO 模型，根据不同的版本可能包含不同数量的卷积层、残差块等。
   选择合适的损失函数（如 YOLO 特有的损失函数，考虑边界框位置损失、类别损失等）和优化器进行训练。
   在推理阶段，输入图像，输出检测到的物体的类别和边界框。
