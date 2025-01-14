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
    transforms.Resize((64, 64)),  # 调整为 64x64 尺寸
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize((0.5,), (0.5,))  # 标准化 [-1, 1]
])

# 数据路径
train_dir = "data/train"
test_dir = "data/test"

# 确保数据路径存在
if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise FileNotFoundError("请准备汉字图片数据集并放置在 data/train 和 data/test 文件夹中！")

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 类别映射
print("类别映射（文件夹名称->索引）：", train_dataset.class_to_idx)
idx_to_char = {v: k for k, v in train_dataset.class_to_idx.items()}
print("索引到汉字的映射：", idx_to_char)

# 定义 CNN 模型
class HanziCNN(nn.Module):
    def __init__(self, num_classes):
        super(HanziCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
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

# 如果有保存好的模型，可以在此处尝试加载
model_path = "hanzi_cnn.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print(f"已加载已有模型：{model_path}")
else:
    print("未检测到已训练的模型，将进行训练。")

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练
epochs = 10
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

# 训练完毕后保存模型参数
torch.save(model.state_dict(), model_path)
print(f"模型已保存到：{model_path}")

# 测试
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

# ==================== 新增截屏和标识区域功能 ====================
# 标识框的起始点和结束点
start_point = None
end_point = None
drawing = False

def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point, drawing

    if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键按下
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:  # 鼠标移动
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:  # 鼠标左键抬起
        drawing = False
        end_point = (x, y)

def main():
    """
    主流程：摄像头截屏 -> 标记区域识别 -> 可返回摄像头。
    """
    global start_point, end_point, drawing

    while True:
        # 摄像头截屏部分
        cap = cv2.VideoCapture(0)  # 打开默认摄像头
        if not cap.isOpened():
            print("无法打开摄像头")
            return

        print("按 's' 键截屏，按 'q' 键退出程序")

        screenshot_taken = False
        while not screenshot_taken:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧")
                break

            cv2.imshow("摄像头 - 按 's' 截屏", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # 按 's' 键截屏
                cv2.imwrite("screenshot.jpg", frame)
                print("已截屏并保存为 screenshot.jpg")
                screenshot_taken = True
            elif key == ord('q'):  # 按 'q' 键退出程序
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        cv2.destroyAllWindows()

        # 标记区域部分
        img = cv2.imread("screenshot.jpg")
        if img is None:
            print("未找到截图文件，请先截屏")
            continue

        print("按 'c' 键识别标记区域，按 'r' 键返回摄像头，按 'q' 键退出程序")

        cv2.namedWindow("标记区域")
        cv2.setMouseCallback("标记区域", draw_rectangle)

        while True:
            temp_img = img.copy()
            if start_point and end_point:
                cv2.rectangle(temp_img, start_point, end_point, (0, 255, 0), 2)

            cv2.imshow("标记区域", temp_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):  # 按 'c' 键进行识别
                if start_point and end_point:
                    x1, y1 = start_point
                    x2, y2 = end_point
                    roi = img[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

                    if roi.size == 0:
                        print("未标记有效区域，无法识别")
                    else:
                        cv2.imwrite("temp.jpg", roi)
                        temp_image = Image.open("temp.jpg").convert("L")
                        temp_image = transform(temp_image).unsqueeze(0)

                        model.eval()
                        with torch.no_grad():
                            output = model(temp_image)
                            probs = torch.softmax(output, dim=1)
                            conf, predicted = torch.max(probs, 1)
                            conf = conf.item()
                            predicted_idx = predicted.item()

                        if conf >= 0.76:
                            char = idx_to_char[predicted_idx]
                            print(f"识别结果：{char}，置信度：{conf:.2f}")
                        else:
                            print("没有检测到可识别的汉字")
                else:
                    print("请先用鼠标标记一个区域")
            elif key == ord('r'):  # 按 'r' 键返回摄像头
                cv2.destroyAllWindows()
                break
            elif key == ord('q'):  # 按 'q' 键退出程序
                cv2.destroyAllWindows()
                return

if __name__ == "__main__":
    main()
