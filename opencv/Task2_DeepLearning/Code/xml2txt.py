import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd

sets = ['train', 'test', 'val']
classes = ['cup','battery','waterbottle','modelbattery','drug','pen','brick','stone','bottle']


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open('datas/Annotations/%s.xml' % (image_id), encoding='utf-8')
    out_file = open('datas/cztrain/labels/%s.txt' % (image_id), 'w', encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    if size!= None:
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            # 在这里添加对 'difficult' 元素是否存在的判断逻辑
            obj_difficult = obj.find('difficult')
            if obj_difficult is not None:
                difficult = int(obj_difficult.text)
            else:
                difficult = 0
            cls = obj.find('name').text
            if cls not in classes or difficult == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            print(image_id, cls, b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()
print(wd)
for image_set in sets:
    if not os.path.exists('datas/cztrain/labels/'):
        os.makedirs('datas/cztrain/labels/')
    image_ids = open('datas/imagesset/%s.txt' % (image_set)).read().strip().split()
    list_file = open('datas/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write('datas/cztrain/images/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()