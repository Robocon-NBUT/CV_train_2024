import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# 参数设置
num_classes = 20  # 假设我们有 20 个汉字类别
batch_size = 32
epochs = 65
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 转为灰度图
    transforms.Resize((32, 32)),                 # 调整图片大小
    transforms.RandomRotation(10),  # 随机旋转
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),                       # 转为 Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到 [-1, 1]
])

# 加载数据集
train_dataset = datasets.ImageFolder(root="./dataset/train", transform=transform)
test_dataset = datasets.ImageFolder(root="./dataset/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义类别标签（根据 ImageFolder 自动生成）
class_labels = train_dataset.classes
assert len(class_labels) >= num_classes, f"类别数小于 {num_classes}，请检查数据集。"

# 定义 CNN 模型
class CNN_Char(nn.Module):
    def __init__(self, num_classes):
        super(CNN_Char, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)  # 输出类别数
        )

    def forward(self, x):
        return self.model(x)

# 实例化模型
model = CNN_Char(num_classes=num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练阶段
print("开始训练...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("训练完成！")

# 保存模型
torch.save(model.state_dict(), "cnn_char_model.pth")
print("模型已保存为 cnn_char_model.pth")

# 测试阶段
print("开始测试...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # 输出每张图片的预测类别
        for j in range(len(labels)):
            print(f"Test Image {i * batch_size + j + 1}: True Label = {class_labels[labels[j].item()]}, "
                f"Predicted = {class_labels[predicted[j].item()]}")
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = (correct / total) * 100
print(f"测试准确率：{accuracy:.2f}%")
