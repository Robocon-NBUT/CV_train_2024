import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import math
from torch.utils.tensorboard import SummaryWriter
import datetime
import optuna
from sklearn.cluster import DBSCAN
import logging
import argparse


# 设置日志记录
logging.basicConfig(filename='recognition.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')


# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='Handwritten Chinese Character Recognition')
    parser.add_argument('--train_data_path', type=str, default=r'C:\Users\Elmo\source\repos\SHIBie chinese\train_data',
                        help='Path to the training data')
    parser.add_argument('--test_data_path', type=str, default=r'C:\Users\Elmo\source\repos\SHIBie chinese\test_data',
                        help='Path to the test data')
    parser.add_argument('--model_save_path', type=str, default='handwritten_chinese_model.pth',
                        help='Path to save the model')
    return parser.parse_args()


# 数据预处理，增强数据多样性
def get_transform():
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomRotation(15),
        transforms.RandomCrop((24, 28)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])


# 加载训练集和测试集
def load_datasets(train_data_path, test_data_path, transform):
    train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)
    return train_dataset, test_dataset


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, max_epochs, patience=5):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    patience_counter = 0

    writer = SummaryWriter()

    for epoch in tqdm(range(max_epochs)):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total * 100
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_accuracy, epoch)

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(test_loader)
        val_accuracy = val_correct / val_total * 100
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Accuracy', val_accuracy, epoch)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    writer.close()
    return train_losses, val_losses, train_accuracies, val_accuracies


def objective(trial):
    # 定义超参数搜索空间
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    momentum = trial.suggest_uniform('momentum', 0.1, 0.99)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = HandwritingCNN()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # 使用ReduceLROnPlateau动态调整学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    _, _, _, val_accuracies = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, max_epochs=30)
    return val_accuracies[-1]


# 定义更深的卷积神经网络模型
class HandwritingCNN(nn.Module):
    def __init__(self):
        super(HandwritingCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        self.adaptive_pool = nn.AdaptiveMaxPool2d((3, 3))

        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, len(train_dataset.classes))

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(-1, 128 * 3 * 3)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    args = parse_args()
    TRAIN_DATA_PATH = args.train_data_path
    TEST_DATA_PATH = args.test_data_path
    MODEL_SAVE_PATH = args.model_save_path

    transform = get_transform()
    train_dataset, test_dataset = load_datasets(TRAIN_DATA_PATH, TEST_DATA_PATH, transform)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    print('Best trial:')
    best_trial = study.best_trial
    print('  Value:', best_trial.value)
    print('  Params:')
    for key, value in best_trial.params.items():
        print('    {}: {}'.format(key, value))

    # 使用最佳超参数重新训练模型
    best_lr = best_trial.params['lr']
    best_momentum = best_trial.params['momentum']
    best_batch_size = best_trial.params['batch_size']
    best_weight_decay = best_trial.params['weight_decay']

    train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False)

    model = HandwritingCNN()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=best_lr, momentum=best_momentum, weight_decay=best_weight_decay)
    # 使用ReduceLROnPlateau动态调整学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, train_loader, test_loader, criterion,
                                                                            optimizer, scheduler, max_epochs=30)

    # 绘制损失与准确率曲线
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.show()

    plt.plot(range(len(train_accuracies)), train_accuracies, label='Train Accuracy')
    plt.plot(range(len(val_accuracies)), val_accuracies, label='Val Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Curve')
    plt.legend()
    plt.show()

    # 加载模型用于预测
    model = HandwritingCNN()
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        model.eval()
    except FileNotFoundError:
        print("Model file not found. Please check the path.")
        logging.error("Model file not found. Please check the path.")
        exit()

    def predict_from_camera():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error opening camera.")
            logging.error("Error opening camera.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 图像预处理
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contour_groups = []
            points = []
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    points.append([x + w // 2, y + h // 2])  # 取轮廓中心

            if points:
                clustering = DBSCAN(eps=30, min_samples=1).fit(points)
                labels = clustering.labels_
                for label in np.unique(labels):
                    if label!= -1:
                        group = np.array(points)[labels == label]
                        x_min = group[:, 0].min()
                        y_min = group[:, 1].min()
                        x_max = group[:, 0].max()
                        y_max = group[:, 1].max()

                        handwritten_area = gray[y_min:y_max, x_min:x_max]
                        pil_img = Image.fromarray(handwritten_area)
                        image_tensor = transform(pil_img).unsqueeze(0)

                        with torch.no_grad():
                            outputs = model(image_tensor)
                            probs = nn.functional.softmax(outputs, dim=1)
                            _, predicted = torch.max(outputs.data, 1)
                            confidence = probs[0][predicted.item()].item() * 100

                        result_text = f"{train_dataset.classes[predicted.item()]}: {confidence:.2f}%"
                        # 绘制识别结果矩形框
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        # 绘制识别结果文字，调整位置与字体样式
                        cv2.putText(frame, result_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # 记录日志
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_entry = f"{current_time}: {result_text}"
                        logging.info(log_entry)

            cv2.imshow('Handwritten Character Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'recognition_result_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.jpg', frame)

        cap.release()
        cv2.destroyAllWindows()

    predict_from_camera()
