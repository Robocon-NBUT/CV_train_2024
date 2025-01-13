import cv2
import numpy as np
import os
import tensorflow as tf


# 定义卷积层前向传播
def conv_forward(X, W, b, stride=1, padding=0):
    m, h, w, c = X.shape
    f, _, _, _ = W.shape
    h_out = (h - f + 2 * padding) // stride + 1
    w_out = (w - f + 2 * padding) // stride + 1
    out = np.zeros((m, h_out, w_out, f))
    X_padded = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant')
    for i in range(m):
        for h in range(h_out):
            for w in range(w_out):
                for f in range(W.shape[0]):
                    out[i, h, w, f] = np.sum(X_padded[i, h * stride:h * stride + W.shape[1],
                                                        w * stride:w * stride + W.shape[2], :] * W[f]) + b[f]
    return out


# 定义ReLU激活函数
def relu(X):
    return np.maximum(0, X)


# 定义最大池化层前向传播
def max_pool_forward(X, f=2, stride=2):
    m, h, w, c = X.shape
    h_out = (h - f) // stride + 1
    w_out = (w - f) // stride + 1
    out = np.zeros((m, h_out, w_out, c))
    for i in range(m):
        for h in range(h_out):
            for w in range(w_out):
                for c in range(X.shape[3]):
                    out[i, h, w, c] = np.max(X[i, h * stride:h * stride + f, w * stride:w * stride + f, c])
    return out


# 定义全连接层前向传播
def fc_forward(X, W, b):
    m = X.shape[0]
    out = np.dot(X.reshape(m, -1), W) + b
    return out


# 定义softmax函数
def softmax(X):
    exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)


# 定义交叉熵损失函数
def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
    return loss


# 从文件夹加载图像数据和标签
def load_images_labels_from_folder(folder_path):
    images = []
    labels = []
    label = 0
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            sub_dir = os.path.join(root, dir)
            for file in os.listdir(sub_dir):
                if file.endswith(('.jpg', '.png', '.bmp')):
                    img_path = os.path.join(sub_dir, file)
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            if len(img.shape) == 2:
                                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                            elif len(img.shape) == 3 and img.shape[2] == 2:
                                img = img[:, :, 0]
                                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                            img = np.expand_dims(img, axis=-1)
                            img = cv2.resize(img, (64, 64))
                            img = img / 255.0
                            img = img.astype(np.float32)
                            if img.shape!= (64, 64, 1):
                                raise ValueError(f"Unexpected image shape {img.shape} for {img_path}")
                            images.append(img)
                            labels.append(label)
                        else:
                            print(f"Error reading {img_path}: Could not open image.")
                    except Exception as e:
                        print(f"Error reading {img_path}: {e}")
            label += 1
    return np.array(images), np.array(labels)


# 卷积层反向传播
def conv_backward(dZ, X, W, b, stride=1, padding=0):
    m, h, w, c = X.shape
    f, _, _, _ = W.shape
    h_out = (h - f + 2 * padding) // stride + 1
    w_out = (w - f + 2 * padding) // stride + 1
    dX = np.zeros_like(X)
    dW = np.zeros_like(W)
    db = np.zeros((f,))
    X_padded = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant')
    for i in range(m):
        for h in range(h_out):
            for w in range(w_out):
                for f in range(W.shape[0]):
                    dW[f] += dZ[i, h, w, f] * X_padded[i, h * stride:h * stride + W.shape[1],
                                                        w * stride:w * stride + W.shape[2], :]
                    db[f] += dZ[i, h, w, f]
                    dX_padded = np.zeros_like(X_padded)
                    dX_padded[i, h * stride:h * stride + W.shape[1],
                    w * stride:w * stride + W.shape[2], :] += dZ[i, h, w, f] * W[f]
                    dX += np.pad(dX_padded, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant')
    return dX[:, padding:-padding, padding:-padding, :], dW, db


# 最大池化层反向传播
def max_pool_backward(dZ, X, f=2, stride=2):
    m, h, w, c = X.shape
    h_out = (h - f) // stride + 1
    w_out = (w - f) // stride + 1
    dX = np.zeros_like(X)
    for i in range(m):
        for h in range(h_out):
            for w in range(w_out):
                for c in range(X.shape[3]):
                    pool_slice = X[i, h * stride:h * stride + f, w * stride:w * stride + f, c]
                    mask = pool_slice == np.max(pool_slice)
                    dX[i, h * stride:h * stride + f, w * stride:w * stride + f, c] += dZ[i, h, w, c] * mask
    return dX


# 全连接层反向传播
def fc_backward(dZ, X, W, b):
    m = X.shape[0]
    dX = np.dot(dZ, W.T).reshape(X.shape)
    dW = np.dot(X.reshape(m, -1).T, dZ)
    db = np.sum(dZ, axis=0)
    return dX, dW, db


# 训练模型
def train_model(X_train, y_train, num_epochs=10, batch_size=32, lr=0.001):
    num_classes = len(np.unique(y_train))
    m = X_train.shape[0]
    W1 = np.random.randn(32, 3, 3, 1) * np.sqrt(2. / (3 * 3 * 1))
    b1 = np.zeros((32,))
    W2 = np.random.randn(64, 3, 3, 32) * np.sqrt(2. / (3 * 3 * 32))
    b2 = np.zeros((64,))
    W3 = np.random.randn(64 * 16 * 16, 128) * np.sqrt(2. / (64 * 16 * 16))
    b3 = np.zeros((128,))
    W4 = np.random.randn(128, num_classes) * np.sqrt(2. / 128)
    b4 = np.zeros((num_classes,))

    for epoch in range(num_epochs):
        for i in range(0, m, batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            # 使用tensorflow.keras.utils.to_categorical替代np_utils.to_categorical
            y_batch = tf.keras.utils.to_categorical(y_batch, num_classes)

            # 前向传播
            conv1 = conv_forward(X_batch, W1, b1, padding=1)
            relu1 = relu(conv1)
            pool1 = max_pool_forward(relu1)
            conv2 = conv_forward(pool1, W2, b2, padding=1)
            relu2 = relu(conv2)
            pool2 = max_pool_forward(relu2)
            fc1 = fc_forward(pool2, W3, b3)
            relu3 = relu(fc1)
            logits = fc_forward(relu3, W4, b4)
            y_pred = softmax(logits)

            # 计算损失
            loss = cross_entropy_loss(y_pred, y_batch)

            # 反向传播
            dZ4 = y_pred - y_batch
            dX3, dW4, db4 = fc_backward(dZ4, relu3, W4, b4)
            dZ3 = np.where(relu3 > 0, dX3, 0)
            dX2, dW3, db3 = fc_backward(dZ3, pool2, W3, b3)
            dZ2 = max_pool_backward(dX2, relu2)
            dX1, dW2, db2 = conv_backward(dZ2, pool1, W2, b2, padding=1)
            dZ1 = np.where(relu1 > 0, dX1, 0)
            dX0, dW1, db1 = conv_backward(dZ1, X_batch, W1, b1, padding=1)

            # 更新权重
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
            W3 -= lr * dW3
            b3 -= lr * db3
            W4 -= lr * dW4
            b4 -= lr * db4

            print(f'Epoch {epoch + 1}, Step {i // batch_size}, Loss: {loss}')

    # 保存权重
    np.savez('weights.npz', W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, W4=W4, b4=b4)

    return W1, b1, W2, b2, W3, b3, W4, b4


# 从文件夹加载数据
folder_path = r'D:/Chinese character/train'
images, labels = load_images_labels_from_folder(folder_path)
W1, b1, W2, b2, W3, b3, W4, b4 = train_model(images, labels)