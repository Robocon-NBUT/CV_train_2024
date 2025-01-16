import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import os


# 1. 数据加载与预处理
# 定义数据预处理的流水线
def binarize_image(image):
    # 将图像转换为灰度图
    image = image.convert('L')
    # 获取图像数据
    img_data = image.getdata()
    # 设定阈值，这里使用 128，可根据实际情况调整
    threshold = 140
    # 二值化操作
    binary_data = [0 if p < threshold else 255 for p in img_data]
    # 更新图像数据
    image.putdata(binary_data)
    return image


transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 将图像调整为 28x28 大小
    transforms.Lambda(binarize_image),  # 自定义的二值化操作
    transforms.ToTensor(),  # 将图像转换为 PyTorch 张量，值会被归一化到 [0, 1] 范围
    transforms.Normalize((0.5,), (0.5,))  # 对图像进行标准化处理，使其值分布在 [-1, 1] 范围
])


# 自定义手写汉字数据集类
class ChineseHandwritingDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.data = []
        self.labels = []
        sub_dir = 'train' if train else 'test'
        sub_dir_path = os.path.join(root, sub_dir)
        print(f"Checking sub_dir_path: {sub_dir_path}")
        # 定义数字标签到汉字标签的映射
        label_mapping = {0: '一', 1: '二', 2: '三', 3: '四', 4: '五', 5: '六', 6: '七', 7: '八', 8: '九'}
        for label_dir in os.listdir(sub_dir_path):
            # 将子目录名（汉字）作为原始标签
            original_label = label_dir
            label = list(label_mapping.keys())[list(label_mapping.values()).index(original_label)]
            label_dir_path = os.path.join(sub_dir_path, label_dir)
            print(f"Checking label_dir_path: {label_dir_path}")
            for img_file in os.listdir(label_dir_path):
                img_path = os.path.join(label_dir_path, img_file)
                print(f"Found image file: {img_path}")
                self.data.append(img_path)
                self.labels.append(label)
        print(f"Loaded {len(self.data)} samples for {'train' if train else 'test'} set.")


    def __getitem__(self, index):
        img_path = self.data[index]
        label = self.labels[index]
        image = Image.open(img_path)  # 打开图像
        if self.transform is not None:
            image = self.transform(image)
        return image, label, img_path


    def __len__(self):
        return len(self.data)


# 加载手写汉字数据集
train_dataset = ChineseHandwritingDataset(root=r'C:\Users\谭尧瑞\Desktop\手写汉字数据集', train=True, transform=transform)
test_dataset = ChineseHandwritingDataset(root=r'C:\Users\谭尧瑞\Desktop\手写汉字数据集', train=False, transform=transform)


train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


print(f"Test loader length: {len(test_loader)}")


# 2. 定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 输入 1 通道，输出 32 通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 输入 32 通道，输出 64 通道
        # 定义全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 展平后输入到全连接层
        self.fc2 = nn.Linear(128, 9)  # 9 个类别


    def forward(self, x):
        x = F.relu(self.conv1(x))  # 第一层卷积 + ReLU
        x = F.max_pool2d(x, 2)     # 最大池化
        x = F.relu(self.conv2(x))  # 第二层卷积 + ReLU
        x = F.max_pool2d(x, 2)     # 最大池化
        x = x.view(-1, 64 * 7 * 7)  # 展平
        x = F.relu(self.fc1(x))    # 全连接层 + ReLU
        x = self.fc2(x)            # 最后一层输出
        return x


# 创建模型实例
model = SimpleCNN()


# 3. 定义损失函数与优化器
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)


# 4. 模型训练
num_epochs = 15
model.train()  # 设置模型为训练模式


for epoch in range(num_epochs):
    total_loss = 0
    for images, labels, _ in train_loader:
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失


        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数


        total_loss += loss.item()


    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")


# 5. 模型测试
model.eval()  # 设置模型为评估模式
correct = 0
total = 0


with torch.no_grad():  # 关闭梯度计算
    for images, labels, img_paths in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # 数字标签到汉字标签的映射
        label_mapping = {0: '一', 1: '二', 2: '三', 3: '四', 4: '五', 5: '六', 6: '七', 7: '八', 8: '九'}
        for i in range(len(img_paths)):
            original_label = label_mapping[labels[i].item()]
            predicted_label = label_mapping[predicted[i].item()]
            print(f"Prediction: {predicted_label}")


accuracy = 100 * correct / total if total > 0 else 0  # 避免除以零
print(f"Test Accuracy: {accuracy:.2f}%")