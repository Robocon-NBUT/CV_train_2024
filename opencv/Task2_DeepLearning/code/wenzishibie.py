import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import cv2

# 检查是否有可用的GPU，如果有则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据集类，用于加载指定路径（D:\hanzi）下的手写汉字图像数据
class HandwrittenChineseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化数据集类

        Args:
            root_dir (str): 图像数据所在的根目录路径，这里应为 D:\hanzi，该文件夹下应包含以汉字命名的子文件夹，每个子文件夹里存放对应汉字的手写图片
            transform (callable, optional): 可选的图像预处理转换操作，默认为None
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []
        self.label_to_index = {}

        # 遍历根目录下的所有子文件夹（每个子文件夹代表一个汉字类别）
        for chinese_char_folder in os.listdir(root_dir):
            char_folder_path = os.path.join(root_dir, chinese_char_folder)
            if os.path.isdir(char_folder_path):
                # 遍历当前汉字类别文件夹下的所有图像文件
                for image_file in os.listdir(char_folder_path):
                    if image_file.endswith(('.jpg', '.png')):
                        image_path = os.path.join(char_folder_path, image_file)
                        self.image_files.append(image_path)
                        self.labels.append(chinese_char_folder)

        # 构建汉字标签到数字索引的映射字典
        unique_labels = sorted(set(self.labels))
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self):
        """
        返回数据集的大小，即图像文件的总数
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        根据给定的索引获取图像数据和对应的标签索引

        Args:
            idx (int): 要获取的图像数据的索引

        Returns:
            tuple: 包含处理后的图像数据（张量形式）和对应的标签索引（整数）
        """
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        label_index = self.label_to_index[label]
        if self.transform:
            image = self.transform(image)
        return image, label_index

# 定义卷积神经网络模型
class ChineseCharacterCNN(nn.Module):
    def __init__(self, num_classes):
        super(ChineseCharacterCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 32 * 16 * 16)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# 数据预处理操作
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 指定图像数据所在的根目录路径为 D:\hanzi
data_root_dir = 'D:\\hanzi'
# 创建数据集实例
dataset = HandwrittenChineseDataset(data_root_dir, transform=transform)

# 划分训练集和测试集，这里按照8:2的比例划分
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# 创建训练集和测试集的数据加载器，设置批量大小等参数
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 实例化模型，假设要识别20个汉字，根据实际情况调整num_classes
model = ChineseCharacterCNN(19)
# 将模型转移到指定的设备（GPU或CPU）上
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        # 将输入数据（图像和标签）也转移到指定的设备（GPU或CPU）上
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader):.4f}')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # 将输入数据（图像和标签）也转移到指定的设备（GPU或CPU）上
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {correct / total:.4f}')

# 保存训练好的模型
torch.save(model.state_dict(), 'chinese_character_model.pth')

# 加载训练好的模型
model.load_state_dict(torch.load('chinese_character_model.pth'))
model.eval()

# 调用摄像头拍照并进行识别
cap = cv2.VideoCapture(0)  # 打开默认摄像头（参数0表示默认摄像头，可根据实际情况调整）
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    ret, frame = cap.read()  # 读取摄像头帧
    if not ret:
        print("无法获取摄像头帧")
        break

    # 将摄像头获取的图像从BGR格式转换为RGB格式（因为后续处理需要RGB格式）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    input_tensor = transform(pil_image).unsqueeze(0)
    # 将输入数据（图像张量）转移到指定的设备（GPU或CPU）上
    input_tensor = input_tensor.to(device)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # 按下空格键拍照并进行识别
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_index = torch.max(output.data, 1)
            predicted_label = list(dataset.label_to_index.keys())[predicted_index.item()]
            print(f"预测结果为汉字：{predicted_label}")

    # 显示摄像头画面，可根据需求选择是否显示，这里显示是为了让用户看到实时画面
    cv2.imshow('Handwritten Chinese Character Recognition', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 按下'q'键退出循环
        break

cap.release()
cv2.destroyAllWindows()