import xml.etree.ElementTree as ET
from os import getcwd
import os
from os.path import join

sets = ['train', 'test', 'val']  # 划分的训练、验证、测试集
classes = ['其他垃圾', '可回收垃圾', '有害垃圾']  # 类别名称


# 进行归一化操作
def convert(size, box):  # size: (原图w, 原图h) , box: (xmin, xmax, ymin, ymax)
    dw = 1. / size[0]  # 1/w
    dh = 1. / size[1]  # 1/h
    x = (box[0] + box[1]) / 2.0  # 物体在图中的中心点x坐标
    y = (box[2] + box[3]) / 2.0  # 物体在图中的中心点y坐标
    w = box[1] - box[0]  # 物体实际像素宽度
    h = box[3] - box[2]  # 物体实际像素高度
    x = x * dw  # 物体中心点x的坐标比(相当于 x/原图w)
    w = w * dw  # 物体宽度的宽度比(相当于 w/原图w)
    y = y * dh  # 物体中心点y的坐标比(相当于 y/原图h)
    h = h * dh  # 物体宽度的宽度比(相当于 h/原图h)
    return (x, y, w, h)  # 返回 相对于原图的物体中心点的x坐标比, y坐标比, 宽度比, 高度比, 取值范围[0-1]


def convert_annotation(image_id):
    """
    将对应文件名的xml文件转化为label文件，xml文件包含了对应的bounding box以及图片长宽大小等信息，
    通过对其解析，进行归一化最终将信息保存到唯一一个label文件中。
    label文件中的格式：class_id x y w h，格式适用于YOLO。
    """
    # 打开xml文件
    in_file = open(f'./Annotations/{image_id}.xml', encoding='utf-8')
    # 准备在对应的image_id中写入对应的label文件
    out_file = open(f'./labels/{image_id}.txt', 'w', encoding='utf-8')

    # 解析xml文件
    tree = ET.parse(in_file)
    root = tree.getroot()

    # 获取图片尺寸
    size = root.find('size')
    if size is not None:
        w = int(size.find('width').text)  # 宽度
        h = int(size.find('height').text)  # 高度

        # 遍历每个object
        for obj in root.findall('outputs/object'):
            for item in obj.findall('item'):
                # 获取类别名称
                cls = item.find('name').text
                # 如果类别不在预定义的类别列表中，则跳过
                if cls not in classes:
                    continue

                # 获取类别的id
                cls_id = classes.index(cls)

                # 获取bounding box坐标
                bndbox = item.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                # 进行归一化操作
                box = (xmin, xmax, ymin, ymax)
                bb = convert((w, h), box)

                # 将类别ID和归一化后的bounding box写入label文件
                out_file.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")

    in_file.close()
    out_file.close()


# 目录设置
image_sets = sets  # 分为训练集、验证集、测试集
base_dir = getcwd()

for image_set in image_sets:
    # 创建 labels 文件夹
    if not os.path.exists('./labels/'):
        os.makedirs('./labels/')

    # 读取图片集列表（train, test, val 等）
    image_ids = open(f'./ImageSets/{image_set}.txt').read().strip().split()

    # 创建对应的 image_set 文件（包含图片路径）
    list_file = open(f'./{image_set}.txt', 'w')

    for image_id in image_ids:
        # 写入图片路径
        list_file.write(f'./images/{image_id}.jpg\n')  # 图片路径
        # 转换 annotation 并保存为 label 文件
        convert_annotation(image_id)

    list_file.close()
