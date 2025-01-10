import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torchvision import datasets, transforms
import cv2


class Network(nn.Module):
    def __init__(self):
        self.channel_input = 1
        self.size_kernel = 5
        self.len_padding = 2
        self.channel_c1 = 6
        self.channel_c2 = 16
        self.channel_c3 = 32
        self.channel_c4 = 64
        img_size = 32
        # 重新计算len_flatten
        after_conv1 = (img_size + 2 * self.len_padding - self.size_kernel) // 1 + 1
        after_pool1 = (after_conv1 - 2) // 2 + 1
        after_conv2 = (after_pool1 - self.size_kernel) // 1 + 1
        after_pool2 = (after_conv2 - 2) // 2 + 1
        after_conv3 = (after_pool2 + 2 * 1 - 3) // 1 + 1
        after_pool3 = (after_conv3 - 2) // 2 + 1
        after_conv4 = (after_pool3 + 2 * 1 - 3) // 1 + 1
        after_pool4 = (after_conv4 - 2) // 2 + 1
        self.len_flatten = self.channel_c4 * after_pool4 * after_pool4
        self.len_hidden = 256
        self.len_out = 1000
        super(Network, self).__init__()
        self.c1 = nn.Conv2d(self.channel_input, self.channel_c1, kernel_size=self.size_kernel, padding=self.len_padding)
        self.c2 = nn.Conv2d(self.channel_c1, self.channel_c2, kernel_size=self.size_kernel)
        self.c3 = nn.Conv2d(self.channel_c2, self.channel_c3, kernel_size=3, padding=1)
        self.c4 = nn.Conv2d(self.channel_c3, self.channel_c4, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(self.len_flatten, self.len_hidden)
        self.fc2 = nn.Linear(self.len_hidden, self.len_out)

        self.learning_rate = 0.001
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        out_c1 = self.c1(x)
        out_sub1 = nn.MaxPool2d(2)(out_c1)
        out_c2 = self.c2(out_sub1)
        out_sub2 = nn.MaxPool2d(2)(out_c2)
        out_c3 = self.c3(out_sub2)
        out_sub3 = nn.MaxPool2d(2)(out_c3)
        out_c4 = self.c4(out_sub3)
        out_sub4 = nn.MaxPool2d(2)(out_c4)
        out_flatten = out_sub4.view(x.size(0), -1)
        out_full_con1 = self.fc1(out_flatten)
        out = self.fc2(out_full_con1)
        return out

    def per_train(self, epoch, train_loader, batch=64, verbose=True, num_view=5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        print('training neural network...')
        for e in range(epoch + 1):
            self.train()
            total_train_loss = 0
            for i, (img, lab) in enumerate(train_loader):
                img, lab = img.to(device), lab.to(device)
                pre = self(img)
                loss_train = nn.CrossEntropyLoss()(pre, lab)
                total_train_loss += loss_train.item()
                loss_train.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            if verbose and e > 0 and e % num_view == 0:
                avg_train_loss = total_train_loss / len(train_loader)
                print(f'epoch: {e}/{epoch} --> training loss: {avg_train_loss}')


def set_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


Seed = 0
set_seed(Seed)

if __name__ == '__main__':
    data_dir = 'D:\\Chnise character'
    img_size = 32
    batch_size = 64

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.ImageFolder(root=data_dir + '/train', transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    net = Network()
    epoch = 150

    net.per_train(epoch, train_loader)

    class_labels = train_dataset.classes

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Camera Input', frame)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_size, img_size))
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        img = transforms.Normalize((0.5,), (0.5,))(img)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device)
        img = img.to(device)

        net.eval()
        with torch.no_grad():
            output = net(img)
            _, predicted = torch.max(output.data, 1)
            if 0 <= predicted.item() < len(class_labels):
                predicted_character = class_labels[predicted.item()]
                print(f"Predicted character: {predicted_character}")
            else:
                print("No character recognized. Continuing...")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()