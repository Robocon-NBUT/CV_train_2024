import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import numpy as np
from torchvision import transforms
a={"shi":"十","tian":"天","yi":"一","bai":"百","hua":"花","huo":"火","mu":"木","ren":"人"}
# 图像预处理函数
def preprocess_image(image):
    # 转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义黑色的范围
    lower_black = np.array([0, 6, 60])
    upper_black = np.array([255, 200, 90])

    # 创建掩码
    mask = cv2.inRange(hsv_image, lower_black, upper_black)

    # 应用掩码
    masked_image = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)

    # 转换为灰度图像
    gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊进行降噪
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # 二值化处理
    _, binary_image = cv2.threshold(blurred_image, 1, 255, cv2.THRESH_BINARY)

    return binary_image


# 自定义数据集类
class HandwrittenChineseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []
        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):  # 确保是文件夹
                self.class_names.append(class_name)
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if os.path.isfile(img_path):  # 确保是文件
                        self.images.append(img_path)
                        self.labels.append(self.class_names.index(class_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图像文件：{img_path}")
        # 应用预处理
        image = preprocess_image(image)
        image = cv2.resize(image, (64, 64))
        image = torch.frombuffer(image.tobytes(), dtype=torch.uint8).view(64, 64, 1).permute(2, 0, 1).float() / 255.0
        label = self.labels[idx]
        return image, label


# 数据准备
data_dir = 'photo'
dataset = HandwrittenChineseDataset(data_dir)

# 检查数据集是否正确加载
if len(dataset) == 0:
    print(f"数据集为空，请检查路径：{data_dir}")
    print("当前目录下的文件和文件夹：")
    for item in os.listdir(data_dir):
        print(item)
    exit(1)

print(f"数据集加载成功，包含 {len(dataset)} 张图像，分为 {len(dataset.class_names)} 个类别。")

# 数据加载器
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 修改输入通道数为1
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, len(dataset.class_names))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# 模型训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 200
early_stopping_threshold = 0.001

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')

    # 检查是否满足早停条件
    if epoch_loss < early_stopping_threshold:
        print(f'Loss低于{early_stopping_threshold}，提前终止训练')
        break

# 模型测试与评估
# 使用摄像头捕获视频流
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

model.eval()
with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break

        # 应用预处理
        frame = preprocess_image(frame)

        # 显示摄像头画面，调整回320x320像素
        frame_display = cv2.resize(frame, (320, 320))
        cv2.imshow('Camera', frame_display)

        # 按 'q' 键截图并进行预测
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # 截图
            captured_image = frame
            # 将截图转换为模型输入格式
            captured_image_resized = cv2.resize(captured_image, (64, 64))
            captured_image_resized = torch.frombuffer(captured_image_resized.tobytes(), dtype=torch.uint8).view(64, 64,
                                                                                                                1).permute(
                2, 0, 1).float() / 255.0
            captured_image_resized = captured_image_resized.unsqueeze(0).to(device)

            # 进行预测
            output = model(captured_image_resized)
            _, predicted = torch.max(output, 1)
            predicted_class = dataset.class_names[predicted.item()]
            print('判断结果:', a[predicted_class])

            # 退出循环
            break

cap.release()
cv2.destroyAllWindows()