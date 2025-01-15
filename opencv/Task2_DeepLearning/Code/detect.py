import cv2
import os
from ultralytics import YOLO

if __name__ == '__main__':
    # 定义输入文件夹的路径
    input_folder = r'D:\Computer Vision\ultralytics\detect'

    # Load a model
    model = YOLO(model=r'D:\Computer Vision\ultralytics\runs\train\rubbish4\weights/best.pt')

    # 获取输入文件夹中的所有图片文件
    image_files = [f for f in os.listdir(input_folder) if
                   os.path.isfile(os.path.join(input_folder, f)) and f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        # 构建完整的输入图像路径
        input_image_path = os.path.join(input_folder, image_file)
        # 读取原始图像
        img = cv2.imread(input_image_path)
        if img is None:
            print(f"Failed to read image: {input_image_path}")
            continue

        # 将图像调整为 320x320 大小
        img_resized = cv2.resize(img, (320, 320))

        # 使用 predict 方法进行预测
        results = model.predict(source=img_resized, save=False, show=False)

        # 获取预测结果中的图像
        img = results[0].plot()

        # 显示图像
        cv2.imshow(f'Detection Result - {image_file}', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("All images processed successfully.")