import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.font_manager import FontProperties


# 设置中文字体
def set_chinese_font():
    # 尝试使用系统中已安装的中文字体
    # 优先级：SimHei > Microsoft YaHei > Noto Sans CJK
    font_paths = [
        'C:\\Windows\\Fonts\\simhei.ttf',  # Windows SimHei
        'C:\\Windows\\Fonts\\msyh.ttc',  # Windows Microsoft YaHei
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'  # Linux Noto Sans CJK
    ]

    font_prop = None
    for path in font_paths:
        if os.path.exists(path):
            font_prop = FontProperties(fname=path)
            break

    if font_prop:
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    else:
        print("未找到指定的中文字体，使用默认字体。中文字符可能无法正确显示。")
        plt.rcParams['axes.unicode_minus'] = False


# 数据转换定义
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# 模型定义
class CNN_Hanzi_Recognizer(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN_Hanzi_Recognizer, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool2(nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.pool3(nn.functional.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# 分类报告函数
def print_classification_report(all_labels, all_preds, classes):
    from collections import defaultdict

    metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for true, pred in zip(all_labels, all_preds):
        if true == pred:
            metrics[true]['tp'] += 1
        else:
            metrics[pred]['fp'] += 1
            metrics[true]['fn'] += 1

    print("\n分类报告:\n")
    for cls_idx, cls in enumerate(classes):
        tp = metrics[cls_idx]['tp']
        fp = metrics[cls_idx]['fp']
        fn = metrics[cls_idx]['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f'类别: {cls}')
        print(f'  精确率: {precision:.4f}')
        print(f'  召回率: {recall:.4f}')
        print(f'  F1分数: {f1_score:.4f}\n')

    correct = sum(1 for t, p in zip(all_labels, all_preds) if t == p)
    total = len(all_labels)
    accuracy = correct / total * 100
    print(f'整体准确率: {accuracy:.2f}%')


# 评估函数
def evaluate_model(model, test_loader, classes, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print_classification_report(all_labels, all_preds, classes)
    visualize_predictions(model, test_loader, classes, device, num_images=10)


# 训练函数
def train_model_with_checkpoint(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_images)
                loss = criterion(val_outputs, val_labels)

                val_loss += loss.item() * val_images.size(0)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = 100 * val_correct / val_total

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'训练损失: {epoch_loss:.4f}, 训练准确率: {epoch_acc:.2f}%, '
              f'验证损失: {val_epoch_loss:.4f}, 验证准确率: {val_epoch_acc:.2f}%')

        # 保存最佳模型
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f'最佳验证准确率: {best_acc:.2f}%')
    model.load_state_dict(best_model_wts)

    return model


# 可视化预测结果
def visualize_predictions(model, test_loader, classes, device, num_images=10):
    set_chinese_font()  # 确保使用中文字体
    model.eval()
    images_shown = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            for i in range(images.size(0)):
                if images_shown >= num_images:
                    return
                img = images[i].cpu().numpy().transpose((1, 2, 0))
                img = (img * 0.5) + 0.5  # 反归一化
                plt.imshow(img.squeeze(), cmap='gray')
                plt.title(f'真实: {classes[labels[i]]}, 预测: {classes[predicted[i]]}')
                plt.axis('off')
                plt.show()
                images_shown += 1


# 单张图像预测（可选）
def predict_image(model, image_path, transform, classes, device):
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0)  # 添加batch维度
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    predicted_class = classes[predicted.item()]
    print(f'预测类别: {predicted_class}')
    return predicted_class


# 打印数据集信息
def print_dataset_info(dataset, name):
    print(f'\n{name} 数据集:')
    print(f'类别数量: {len(dataset.classes)}')
    for cls_idx, cls in enumerate(dataset.classes):
        num_images = len([1 for _, label in dataset.imgs if label == cls_idx])
        print(f'  类别 {cls}: {num_images} 张图像')


# 验证数据集标签
def verify_dataset(dataset):
    for idx, (path, label) in enumerate(dataset.imgs):
        print(f'图像 {idx + 1}: 路径={path}, 标签={label}')


def main():
    print("当前工作目录:", os.getcwd())
    script_dir = os.path.dirname(os.path.abspath(__file__))

    train_path = os.path.join(script_dir, '..', 'dataset', 'train')
    val_path = os.path.join(script_dir, '..', 'dataset', 'val')
    test_path = os.path.join(script_dir, '..', 'dataset', 'test')

    print("训练数据路径:", train_path)
    print("验证数据路径:", val_path)
    print("测试数据路径:", test_path)

    if not os.path.isdir(train_path):
        raise FileNotFoundError(f"训练目录未找到: {train_path}")
    if not os.path.isdir(val_path):
        raise FileNotFoundError(f"验证目录未找到: {val_path}")
    if not os.path.isdir(test_path):
        raise FileNotFoundError(f"测试目录未找到: {test_path}")

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=val_test_transform)

    print_dataset_info(train_dataset, "训练")
    print_dataset_info(val_dataset, "验证")
    print_dataset_info(test_dataset, "测试")

    print("\n验证训练数据集:")
    verify_dataset(train_dataset)
    print("\n验证验证数据集:")
    verify_dataset(val_dataset)
    print("\n验证测试数据集:")
    verify_dataset(test_dataset)

    print("\n类别:", train_dataset.classes)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    num_classes = len(train_dataset.classes)
    model = CNN_Hanzi_Recognizer(num_classes=num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model = train_model_with_checkpoint(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=150)

    torch.save(model.state_dict(), 'cnn_hanzi_recognizer.pth')
    print('最佳模型已保存为 cnn_hanzi_recognizer.pth')

    evaluate_model(model, test_loader, train_dataset.classes, device)


if __name__ == '__main__':
    main()
