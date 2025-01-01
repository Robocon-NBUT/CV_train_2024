import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# 1. 数据集预处理
def preprocess_and_save_images(input_dir, output_dir, image_size=(64, 64)):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历所有子文件夹（每个子文件夹代表一个类别）
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if os.path.isdir(class_dir):
            output_class_dir = os.path.join(output_dir, class_name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)

            # 遍历该类别文件夹中的所有图片
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_name.endswith('.png') or img_name.endswith('.jpg'):
                    # 打开图像
                    img = Image.open(img_path).convert('L')  # 转换为灰度图
                    img = img.resize(image_size)  # 调整图像大小

                    # 保存预处理后的图像
                    img.save(os.path.join(output_class_dir, img_name))
                    print(f"Processed and saved: {os.path.join(output_class_dir, img_name)}")


# 2. 自定义数据集类 (HandwrittenHanziDataset)
class HandwrittenHanziDataset(Dataset):
    def __init__(self, data_dir, transform=None, image_size=(64, 64)):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.image_size = image_size  # 新增的 image_size 属性

        # 遍历每个类别文件夹（每个文件夹对应一个汉字）
        for label, class_name in enumerate(os.listdir(data_dir)):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.png') or img_name.endswith('.jpg'):  # 图像格式
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(label)  # 每个文件夹的名称即为标签（汉字）

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('L')  # 将图像转换为灰度图
        image = image.resize(self.image_size)  # 将图像调整为指定的大小

        if self.transform:
            image = self.transform(image)

        return image, label


# 3. 图像预处理和加载
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 数据增强：水平翻转
    transforms.RandomRotation(20),  # 数据增强：随机旋转
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 对图像进行归一化，保持与训练时一致
])

# 假设你将图像分辨率设置为 64x64
train_set = HandwrittenHanziDataset(data_dir=r"D:\黄世昌\Desktop\handwritten_data\train", transform=transform,
                                    image_size=(64, 64))
test_set = HandwrittenHanziDataset(data_dir=r"D:\黄世昌\Desktop\handwritten_data\test", transform=transform,
                                   image_size=(64, 64))

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)


# 4. 定义神经网络模型
class HandwrittenHanziModel(nn.Module):
    def __init__(self, num_classes=20):  # num_classes 参数可以根据需要调整
        super(HandwrittenHanziModel, self).__init__()
        self.convl1 = nn.Conv2d(1, 32, 3, padding=1)  # 卷积层，输入通道1（灰度图），输出通道32，卷积核大小3
        self.convl2 = nn.Conv2d(32, 64, 3, padding=1)  # 第二个卷积层
        self.pool = nn.MaxPool2d(2, 2)  # 池化层，大小为2x2
        # 修改全连接层输入大小，假设图像大小为64x64
        self.fc1 = nn.Linear(64 * 16 * 16, 256)  # 更新为新的维度
        self.fc2 = nn.Linear(256, num_classes)  # 输出层，假设有num_classes个汉字类别

    def forward(self, x):
        x = self.pool(F.relu(self.convl1(x)))  # 卷积+ReLU激活+池化
        x = self.pool(F.relu(self.convl2(x)))  # 第二个卷积+ReLU激活+池化
        x = x.view(-1, 64 * 16 * 16)  # 扁平化操作
        x = F.relu(self.fc1(x))  # 第一个全连接层
        x = self.fc2(x)  # 第二个全连接层
        return F.log_softmax(x, dim=1)  # 使用log_softmax作为输出


# 5. 定义训练过程
def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 部署到设备上（GPU或CPU）
        optimizer.zero_grad()  # 梯度初始化为0
        output = model(data)  # 模型预测
        loss = F.cross_entropy(output, target)  # 计算交叉熵损失
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器更新参数

        total_loss += loss.item() * data.size(0)  # 累计损失
        pred = output.argmax(dim=1, keepdim=True)  # 取最大值的索引作为预测结果
        correct += pred.eq(target.view_as(pred)).sum().item()  # 累计正确的预测数量

        if batch_idx % 100 == 0:  # 每100个batch输出一次训练信息
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

    train_loss = total_loss / len(train_loader.dataset)
    train_accuracy = 100. * correct / len(train_loader.dataset)
    print(f"Train Epoch {epoch}: Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")


# 6. 定义测试过程
def test_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 不计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # 计算总损失
            pred = output.argmax(dim=1, keepdim=True)  # 取最大值的索引作为预测结果
            correct += pred.eq(target.view_as(pred)).sum().item()  # 计算正确的预测数量

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")


# 7. 模型保存和加载
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Model loaded from {filepath}")


# 8. 主程序 - 训练和测试模型
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = HandwrittenHanziModel(num_classes=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

    epochs = 200  # 可以在此修改训练轮数
    for epoch in range(1, epochs + 1):
        train_model(model, device, train_loader, optimizer, epoch)
        test_model(model, device, test_loader)

    save_model(model, "hanzi_model.pth")


# 9. 预测图像
def predict_image(model, device, image_path, class_names):
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 与训练时一致
    ])

    image = Image.open(image_path).convert('L')
    image = image.resize((64, 64))
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True)
        predicted_label = pred.item()
        predicted_char = class_names[predicted_label]  # 获取对应的汉字
        print(f"Predicted character: {predicted_char}")


# 10. 运行代码
if __name__ == "__main__":
    # 预处理和保存图像
    preprocess_and_save_images(r"D:\黄世昌\Desktop\handwritten_data\train", r"D:\黄世昌\Desktop\handwritten_data\processed_train")
    preprocess_and_save_images(r"D:\黄世昌\Desktop\handwritten_data\test", r"D:\黄世昌\Desktop\handwritten_data\processed_test")

    # 加载训练好的模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HandwrittenHanziModel(num_classes=20).to(device)
    load_model(model, "hanzi_model.pth")

    # 获取类别（汉字）名称
    class_names = sorted(os.listdir(r"D:\黄世昌\Desktop\handwritten_data\train"))

    # 测试预测
    predict_image(model, device, r"D:\黄世昌\Desktop\handwritten_data\processed_train\一\一_55.png", class_names)
