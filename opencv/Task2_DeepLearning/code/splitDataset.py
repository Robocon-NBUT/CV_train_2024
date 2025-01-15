import os
import random

# XML标注文件所在路径，使用绝对路径
xmlfilepath = 'D:/yolo_data/Annotations'
# 记录数据集划分结果的文件保存路径，修改为绝对路径 D:\yolo_data\ImageSets
txtsavepath = 'D:/yolo_data/ImageSets'

# 检查标注文件所在文件夹是否存在，如果不存在则抛出异常并提示用户
if not os.path.exists(xmlfilepath):
    raise FileNotFoundError(f"The Annotations folder at {xmlfilepath} does not exist. Please check the path and make sure the XML files are located there.")

# 检查保存数据集划分结果的文件夹是否存在，不存在则创建该文件夹
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)

trainval_percent = 0.9
train_percent = 0.9

tv = int(num * trainval_percent)
tr = int(tv * train_percent)

trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

# 使用绝对路径打开各个用于记录数据集划分结果的文件
ftrainval = open(os.path.join(txtsavepath, 'trainval.txt'), 'w')
ftest = open(os.path.join(txtsavepath, 'test.txt'), 'w')
ftrain = open(os.path.join(txtsavepath, 'train.txt'), 'w')
fval = open(os.path.join(txtsavepath, 'val.txt'), 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
