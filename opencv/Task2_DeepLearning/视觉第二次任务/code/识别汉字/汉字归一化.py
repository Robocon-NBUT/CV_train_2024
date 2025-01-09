from PIL import Image
import os


def normalize_images(input_folder, output_folder, size=(128, 128)):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            # 打开图像文件
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)

            # 转换为灰度图像
            img = img.convert('L')

            # 调整图像大小
            img = img.resize(size, Image.LANCZOS)

            # 保存归一化后的图像到输出文件夹
            img.save(os.path.join(output_folder, filename))


# 使用函数
input_folder = r'C:\Users\黄广松\PyCharmMiscProject\pytorch\汉字识别\train汉字\test_image.png'  # 输入文件夹路径
output_folder = r'C:\Users\黄广松\PyCharmMiscProject\pytorch\汉字识别\train汉字\test_image.png'  # 输出文件夹路径
normalize_images(input_folder, output_folder)