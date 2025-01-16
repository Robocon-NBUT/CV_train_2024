import os
import random
trainval_percent=0.9
train_percent=0.9
xmlfilepath= './datas/Annotations'
txtsavepath= './datas/images'
total_xml=os.listdir(xmlfilepath)

num=len(total_xml)
list=range(num)
tv=int(num * trainval_percent)
tr=int(tv * train_percent)
trainval=random.sample(list,tv)
train=random.sample(trainval,tr)
output_dir = './datas/imagesset/'

# 确保目标文件夹存在，若不存在则创建
os.makedirs(output_dir, exist_ok=True)  # exist_ok=True: 如果目录已经存在，则不抛出异常

# 打开文件进行写操作
ftrainval = open(os.path.join(output_dir, 'trainval.txt'), 'w')
ftrainval=open('./datas/imagesset/trainval.txt','w')
ftest=open('./datas/imagesset/test.txt','w')
ftrain=open('./datas/imagesset/train.txt','w')
fval=open('./datas/imagesset/val.txt','w')

for i in list:
    name=total_xml[i][:-4]+'\n'
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