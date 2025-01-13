import xml.etree.ElementTree as ET
import os


# 定义数据集的划分，包括训练集、测试集和验证集
sets = ['train', 'test', 'val']
# 定义目标检测的类别列表
classes = ['drug', 'cup', 'bottle', 'battery', 'modelbattery','bag','plastic','rock','stone']


def convert(size, box):
    """
    将边界框从原始坐标转换为相对于图像尺寸的归一化坐标。

    参数:
    size (tuple): 图像的宽度和高度 (w, h)
    box (tuple): 边界框的坐标 (xmin, xmax, ymin, ymax)

    返回:
    tuple: 转换后的边界框坐标 (x, y, w, h)，其中 (x, y) 是边界框的中心坐标，(w, h) 是宽度和高度
    """
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
    """
    将给定图像ID的XML标注文件转换为YOLO格式的文本标注文件。

    参数:
    image_id (str): 图像的唯一标识符

    返回:
    None
    """
    xml_file_path = f'D:/pytorch/deep learning/yolo/my_datasets/Annotations/{image_id}.xml'
    label_file_path = f'D:/pytorch/deep learning/yolo/my_datasets/labels/{image_id}.txt'

    # 检查XML文件是否存在
    if not os.path.exists(xml_file_path):
        print(f'Warning: {xml_file_path} does not exist. Skipping...')
        return

    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        size = root.find('size')
        if size:
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            with open(label_file_path, 'w', encoding='utf-8') as out_file:
                for obj in root.iter('object'):
                    difficult_elem = obj.find('difficult')
                    difficult = '0' if difficult_elem is None else difficult_elem.text
                    cls = obj.find('name').text
                    # 检查类别是否在预定义的类别列表中，并且是否标记为困难样本
                    if cls not in classes or int(difficult) == 1:
                        continue
                    cls_id = classes.index(cls)
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                         float(xmlbox.find('ymax').text))
                    print(image_id, cls, b)
                    bb = convert((w, h), b)
                    out_file.write(f'{cls_id} {" ".join(map(str, bb))}\n')
    except Exception as e:
        print(f'Error processing {xml_file_path}: {e}')


wd = os.getcwd()
print(wd)

# 遍历每个数据集划分（训练集、测试集、验证集）
for image_set in sets:
    labels_dir = 'D:/pytorch/deep learning/yolo/my_datasets/labels'
    # 如果标签目录不存在，则创建它
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    image_sets_file = f'D:/pytorch/deep learning/yolo/my_datasets/ImageSets/{image_set}.txt'
    # 检查图像集文件是否存在
    if not os.path.exists(image_sets_file):
        print(f'Warning: {image_sets_file} does not exist. Skipping {image_set} set...')
        continue

    list_file_path = f'D:/pytorch/deep learning/yolo/my_datasets/{image_set}.txt'
    with open(list_file_path, 'w', encoding='utf-8') as list_file:
        with open(image_sets_file, 'r', encoding='utf-8') as f:
            image_ids = f.read().strip().split()
            # 遍历图像集中的每个图像ID
            for image_id in image_ids:
                list_file.write(f'D:/pytorch/deep learning/yolo/my_datasets/images/{image_id}.jpg\n')
                convert_annotation(image_id)
