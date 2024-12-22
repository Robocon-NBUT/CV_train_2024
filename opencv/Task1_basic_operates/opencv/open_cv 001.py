import cv2
import os

# 获取当前脚本所在的目录
script_directory = os.path.dirname(os.path.abspath(__file__))

# 创建输出文件夹，保存在脚本所在目录中
output_folder = os.path.join(script_directory, 'captured_images')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 打开摄像头
cap = cv2.VideoCapture(0)

print("按下 's' 键拍摄并保存图片为 all.jpg，按 'q' 键退出。")

while True:
    # 读取摄像头画面
    ret, frame = cap.read()

    # 显示摄像头画面
    cv2.imshow('Video', frame)

    # 检测按键
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 按下 'q' 键退出
        break
    elif key == ord('s'):  # 按下 's' 键拍照并保存
        all_path = os.path.join(output_folder, 'all.jpg')
        cv2.imwrite(all_path, frame)
        print(f"图片已保存为: {all_path}")

cap.release()
cv2.destroyAllWindows()
