import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import logging
import cv2
from PIL import Image


# 设置日志记录的基本配置，日志级别为INFO，格式为时间-日志级别-消息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#CNN神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(3,16,kernel_size=5,stride=1,padding=2)
        self.relu1=nn.ReLU()
        self.poll1=nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv2=nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1)
        self.relu2=nn.ReLU()
        self.poll2=nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3=nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.relu3=nn.ReLU()
        self.poll3=nn.MaxPool2d(kernel_size=2,stride=2)

        self.fc=nn.Linear(64*64*64,22)
    def forward(self,x):
        x=self.conv1(x)
        x=self.relu1(x)
        x=self.poll1(x)

        x=self.conv2(x)
        x=self.relu2(x)
        x=self.poll2(x)

        x=self.conv3(x)
        x=self.relu3(x)
        x=self.poll3(x)

        x=x.view(x.size(0),-1)
        x=self.fc(x)

        return x

# 数据预处理
def get_data_transform(train=False):
    return transforms.Compose([
    transforms.RandomRotation(degrees=(-30, 30)),
    transforms.Resize((512, 512)),
    # 去掉Grayscale转换
    transforms.ToTensor(),            
    # 调整归一化参数以适应3通道图像
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 从摄像头获取图像并进行预测
def predict_from_camera(model, class_names):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        error = cv2.error()
        logging.error(f"无法打开摄像头: {error}")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("无法获取摄像头图像")
                break

            frame = cv2.resize(frame, (640, 480))
            cv2.imshow('Take a picture of handwritten Chinese character', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                original_img = img.copy()
                img = Image.fromarray(img)
                data_transform = get_data_transform()
                img = data_transform(img)
                img = img.unsqueeze(0)

                model.eval()
                with torch.no_grad():
                    outputs = model(img)
                    _, predicted = torch.max(outputs.data, 1)
                    predicted_class = class_names[predicted.item()]
                    logging.info(f'预测的汉字是: {predicted_class}')

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (10, 30)
                    fontScale = 1
                    fontColor = (0, 255, 0)
                    lineType = 2
                    cv2.putText(original_img, f'预测: {predicted_class}', bottomLeftCornerOfText, font,
                                fontScale, fontColor, lineType)

                    original_img = cv2.resize(original_img, (640, 480))
                    cv2.imshow('Prediction Result', original_img)
            elif key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


# 加载模型
def load_trained_model(model_path):
    try:
        model = CNN()
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict)
        return model
    except FileNotFoundError as e:
        logging.error(f"模型文件 {model_path} 未找到: {e}")
        raise
    except Exception as e:
        logging.error(f"加载模型时出错: {e}")
        raise


# 模型路径和类别名称
model_path = 'D:/pytorch/deep learning/parameter/saved_model_state.pth'
class_names = ["爱", "不", "大", "地", "的", "风", "火", "极", "林", "们",
               "人", "日", "山", "上", "水", "天", "我", "下", "有", "月",
               "在", "中"]

model = load_trained_model(model_path)
predict_from_camera(model, class_names)



