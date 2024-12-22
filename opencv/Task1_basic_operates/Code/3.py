import cv2
all_img = cv2.imread('all.jpg')
if all_img is None:
    print("无法打开 all.jpg，请检查文件路径。")
    exit()
phone_img = cv2.imread('phone.jpg')
if phone_img is None:
    print("无法打开 all.jpg，请检查文件路径。")
    exit()
all_height, all_width = all_img.shape[:2]
phone_resized = cv2.resize(phone_img, (all_width, all_height), interpolation=cv2.INTER_AREA)
save_path = 'phone_resized.jpg'
cv2.imwrite(save_path, phone_resized)
print(f"已将 phone.jpg 缩放至与 all.jpg 一致的尺寸，并保存为 {save_path}")