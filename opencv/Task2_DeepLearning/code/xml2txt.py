# -*- coding: utf-8 -*-
# xml解析包
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

# 定义数据集划分的几个集合，通常包含训练集、测试集、验证集等
sets = ['train', 'test', 'val']
# 定义数据集中包含的类别列表，需根据实际标注情况填写准确的类别名称
classes = ['drug', 'cup', 'bottle', 'battery','bag','plastic','rock','stone','modelbattery']


# 进行归一化操作
def convert(size, box):
    """
    该函数用于将标注框的坐标信息根据图片的尺寸进行归一化处理。

    参数:
    size: 包含原图宽和高的元组，格式为 (原图w, 原图h)
    box: 包含标注框坐标信息的元组，格式为 (xmin, xmax, ymin, ymax)

    返回值:
    经过归一化后的标注框坐标信息元组，格式为 (x坐标比, y坐标比, 宽度比, 高度比)，取值范围在 [0 - 1]
    """
    dw = 1. / size[0]  # 计算原图宽度的倒数，即 1 / w
    dh = 1. / size[1]  # 计算原图高度的倒数，即 1 / h
    x = (box[0] + box[1]) / 2.0  # 计算物体在图中的中心点x坐标
    y = (box[2] + box[3]) / 2.0  # 计算物体在图中的中心点y坐标
    w = box[1] - box[0]  # 计算物体实际像素宽度
    h = box[3] - box[2]  # 计算物体实际像素高度
    x = x * dw  # 将物体中心点x坐标转换为相对于原图宽度的坐标比（相当于 x / 原图w）
    w = w * dw  # 将物体宽度转换为相对于原图宽度的宽度比（相当于 w / 原图w）
    y = y * dh  # 将物体中心点y坐标转换为相对于原图高度的坐标比（相当于 y / 原图h）
    h = h * dh  # 将物体高度转换为相对于原图高度的高度比（相当于 h / 原图h）
    return (x, y, w, h)  # 返回归一化后的坐标信息


def convert_annotation(image_id):
    """
    将对应文件名的xml文件转化为label文件。xml文件包含了对应的bounding框以及图片长宽大小等信息，
    通过对其解析，然后进行归一化最终将信息保存到label文件中。即一张图片文件对应一个xml文件，
    经过解析和归一化后，将对应的信息保存到唯一的label文件中。
    label文件中的格式：class x y w h ，同时，一张图片对应的类别可能有多个，所以对应的bounding的信息也有多个。

    参数:
    image_id: 图片对应的文件名（不含扩展名），用于定位对应的xml文件和生成对应的label文件。
    """
    # 使用绝对路径打开对应的xml文件，假设xml文件存放在 D:\yolo_data\Annotations 目录下
    in_file = open('D:/yolo_data/Annotations/%s.xml' % (image_id), encoding='utf-8')
    # 使用绝对路径创建对应的label文件，用于保存解析和归一化后的标注信息，假设保存到 D:\yolo_data\labels 目录下
    out_file = open('D:/yolo_data/labels/%s.txt' % (image_id), 'w', encoding='utf-8')
    # 解析xml文件
    tree = ET.parse(in_file)
    # 获得xml文件的根节点，方便后续查找各种标签信息
    root = tree.getroot()
    # 获得图片的尺寸大小信息
    size = root.find('size')
    # 如果xml内的标记为空，增加判断条件，避免出现属性获取报错
    if size is not None:
        # 获得图片宽度信息
        w = int(size.find('width').text)
        # 获得图片高度信息
        h = int(size.find('height').text)
        # 遍历xml文件中所有的 'object' 标签，每个 'object' 标签通常对应一个标注目标
        for obj in root.iter('object'):
            # 获得 'difficult' 属性的值，其含义可能与标注的难易程度等相关（具体需根据数据集定义判断）
            difficult = obj.find('difficult').text
            # 获得标注目标的类别名称，为字符串类型
            cls = obj.find('name').text
            # 如果类别不在预定义的类别列表中，或者 'difficult' 属性值为 1，则跳过该目标的处理
            if cls not in classes or int(difficult) == 1:
                continue
            # 通过类别名称在类别列表中找到对应的类别id
            cls_id = classes.index(cls)
            # 找到 'bndbox' 对象，其下包含了标注框的坐标信息
            xmlbox = obj.find('bndbox')
            # 获取对应的bndbox的坐标信息数组，格式为 ['xmin', 'xmax', 'ymin', 'ymax']，并转换为浮点数类型
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            # 调用convert函数对坐标信息进行归一化处理
            bb = convert((w, h), b)
            # 将类别id和归一化后的坐标信息写入到label文件中，格式为 "class_id x y w h"，每个目标信息占一行
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


# 获取当前工作目录（在原代码中可能有其他用途，但这里主要为了展示完整代码结构保留了这部分）
wd = getcwd()
print(wd)

# 对所有的文件数据集进行遍历，主要完成两个工作：
# 1. 将所有图片文件的全路径都写在对应的txt文件中去，方便定位图片。
# 2. 同时对所有的图片文件对应的xml文件进行解析和转化，将其对应的bounding box以及类别的信息全部解析写到label文件中去，
#    最后通过直接读取文件，就能找到对应的label信息。
for image_set in sets:
    # 先检查 D:\yolo_data\labels 文件夹是否存在，如果不存在则创建该文件夹，用于保存生成的label文件
    if not os.path.exists('D:/yolo_data/labels/'):
        os.makedirs('D:/yolo_data/labels/')
    # 使用绝对路径读取在 D:\yolo_data\ImageSets 中的train、test、val等文件的内容，这些文件包含对应的图片文件名（不含扩展名）
    image_ids = open('D:/yolo_data/ImageSets/%s.txt' % (image_set)).read().strip().split()
    # 使用绝对路径打开对应的数据集划分文件（如 D:\yolo_data\train.txt 等），用于写入图片文件路径信息等，准备写入操作
    list_file = open('D:/yolo_data/%s.txt' % (image_set), 'w')
    # 遍历每个图片文件名（不含扩展名）
    for image_id in image_ids:
        # 将图片文件的绝对路径信息写入到对应的数据集划分文件中，格式为 "D:/yolo_data/images/image_id.png"，每个路径占一行
        list_file.write('D:/yolo_data/images/%s.jpg\n' % (image_id))
        # 调用convert_annotation函数，将对应的xml文件解析并生成label文件
        convert_annotation(image_id)
    # 关闭打开的数据集划分文件，释放资源
    list_file.close()
