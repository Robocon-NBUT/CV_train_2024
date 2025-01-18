import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2
from PIL import Image
import os


# 设置随机种子
torch.manual_seed(42)

# 数据集预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 转为灰度图
    transforms.Resize((32, 32)),  # 调整为 32x32 尺寸
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize((0.5,), (0.5,))  # 标准化 [-1, 1]
])

# 加载数据集
train_dir = "picture/train"  # 训练集文件夹路径
test_dir = "picture/test"  # 测试集文件夹路径

# 确保数据路径存在
if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise FileNotFoundError("请准备汉字图片数据集并放置在 picture/train 和 picture/test 文件夹中！")

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=18, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=18, shuffle=False)

# 打印类别映射
print("类别映射（文件夹名称->索引）：", train_dataset.class_to_idx)

# 定义类别索引到汉字的映射
# 示例：假设类别文件夹名称为“汉字1”,“汉字2”，则需要手动定义映射
idx_to_char = {v: k for k, v in train_dataset.class_to_idx.items()}
print("索引到汉字的映射：", idx_to_char)

# 定义 CNN 模型
class HanziCNN(nn.Module):
    def __init__(self, num_classes):
        super(HanziCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 获取类别数量
num_classes = len(train_dataset.classes)

# 实例化模型
model = HanziCNN(num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

# 保存训练后的模型
torch.save(model.state_dict(), "hanzi_model.pth")  # 保存模型参数到文件

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"测试准确率: {correct / total * 100:.2f}%")


# 预测单个汉字
def predict_single_image(image_path):
    # 加载已保存的模型参数
    model = HanziCNN(num_classes)
    model.load_state_dict(torch.load("hanzi_model.pth", weights_only=True))
    model.eval()

    # 加载并预处理图片
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0)  # 增加 batch 维度

    # 推理
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()

    # 输出结果
    char = idx_to_char[class_idx]
    print(f"预测结果：{char}")


# 打开摄像头并进行汉字识别（整合后的函数）
def predict_from_camera():
    # 加载已保存的模型参数
    model = HanziCNN(num_classes)
    model.load_state_dict(torch.load("hanzi_model.pth", weights_only=True))
    model.eval()

    cap = cv2.VideoCapture(0)  # 打开默认摄像头

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("按 'esc' 键退出摄像头，按 'a' 键选择文字区域并识别")
    roi_box = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧")
            break

        # 绘制选择框
        if roi_box is not None:
            cv2.rectangle(frame, (roi_box[0], roi_box[1]), (roi_box[0] + roi_box[2], roi_box[1] + roi_box[3]), (0, 255, 0), 2)

        # 显示摄像头画面
        cv2.imshow('摄像头 - 请书写汉字', frame)

        # 按 'a' 键进行预测
        key = cv2.waitKey(1)
        if key & 0xFF == ord('a'):
            # 选择文字区域
            roi_box = cv2.selectROI('摄像头 - 请选择文字区域', frame, False)
            # 裁剪选择的区域
            roi_frame = frame[int(roi_box[1]):int(roi_box[1] + roi_box[3]), int(roi_box[0]):int(roi_box[0] + roi_box[2])]
            # 保存临时图片
            cv2.imwrite("temp.jpg", roi_frame)
            # 加载并预处理图片
            temp_image = Image.open("temp.jpg").convert("L")
            temp_image = transform(temp_image).unsqueeze(0)

            # 推理
            with torch.no_grad():

                output = model(temp_image)
                _, predicted = torch.max(output, 1)
                class_idx = predicted.item()

            char = idx_to_char[class_idx]
            print(f"识别结果：{char}")

        # 按 'q' 键退出
        k = cv2.waitKey(1)
        if k == 27:
            # 通过esc键退出摄像
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()


# 示例：打开摄像头
predict_from_camera()
