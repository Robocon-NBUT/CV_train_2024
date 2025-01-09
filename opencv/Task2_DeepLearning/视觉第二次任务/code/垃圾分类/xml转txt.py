import os
import re
from tqdm import tqdm
classes = []
classes_dict = {}
def convert(Imgsize, box):
    width, height = Imgsize
    xmin, ymin, xmax, ymax = box
    dw = max(xmin, xmax) - min(xmin, xmax)
    dh = max(ymin, ymax) - min(ymin, ymax)
    x_ = (xmin + xmax) / 2.0
    y_ = (ymin + ymax) / 2.0
    cx = round(x_ / width, 6)
    cy = round(y_ / height, 6)
    w = round(dw / width, 6)
    h = round(dh / height, 6)
    return [cx, cy, w, h]

def save_txt(namepath, text):
    with open(namepath, 'w') as f:
        f.write(text)

def convert_annotation(xml_path, name):
    xml_name = os.path.join(xml_path, name)
    with open(xml_name, "r", encoding="utf-8") as f1:
        text = f1.read().replace("\n", "")
        text = text.replace(" ", "")
    img_size = re.findall("<width>([0-9]+)</width>.*?<height>([0-9]+)</height>", text)[0]
    # find_datas = re.findall(
    #     "<object>.*?<name>([a-z|A-Z]*?)</name>.*?<xmin>([0-9]+?)</xmin>.*?<ymin>([0-9]+?)</ymin>.*?<xmax>([0-9]+?)</xmax>.*?<ymax>([0-9]+?)</ymax>",
    #     text)
    find_datas = re.findall(
        "<object>.*?<name>([^<]+)</name>.*?<xmin>([0-9]+?)</xmin>.*?<ymin>([0-9]+?)</ymin>.*?<xmax>([0-9]+?)</xmax>.*?<ymax>([0-9]+?)</ymax>",
        text)

    savetext = ""
    for item in find_datas:
        class_ = item[0]
        if class_ not in classes:
            classes.append(class_)
            classes_dict[class_] = len(classes) - 1

        imgsize = [int(img_size[0]), int(img_size[1])]
        box = [int(item[1]), int(item[2]), int(item[3]), int(item[4])]
        site = convert(imgsize, box)
        savetext += "{0} {1} {2} {3} {4}".format(classes_dict[class_], site[0], site[1], site[2], site[3])
        savetext += "\n"
    name = name.split(".")[0]
    save_txt(labels_p + "/" + name + ".txt", savetext.strip())
    # print(classes_dict)
    # print(classes)
if __name__ == "__main__":
    # root_path = os.getcwd()
    xml_DIR = r"D:\Users\黄广松\Desktop\ultralytics-8.3.2\垃圾识别\三合一\三合一XML"
    OUTPUT_DIR = r"D:\Users\黄广松\Desktop\ultralytics-8.3.2\垃圾识别\三合一\三合一TXT"

    labels_p = os.path.join(OUTPUT_DIR)  # txt保存输出路径
    try:
        os.makedirs(labels_p)
    except:
        pass
    xml_path = os.path.join(xml_DIR)  # xml的源文件夹
    xml_list = sorted(os.listdir(xml_path))
    #进度条
    total_files = len(xml_list)
    with tqdm(total=total_files, desc="Processing XML Files") as pbar:
        for name in xml_list:
            convert_annotation(xml_path, name)
            pbar.update(1)
