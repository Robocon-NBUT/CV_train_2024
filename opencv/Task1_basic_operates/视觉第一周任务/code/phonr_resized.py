import cv2

# 读取all.jpg图像
img_all = cv2.imread('all.jpg')
if img_all is None:
    print("无法读取all.jpg图像")
    exit()

# 获取all.jpg的尺寸（宽度和高度）
height_all, width_all, _ = img_all.shape

# 读取phone.jpg图像
img_phone = cv2.imread('phone.jpg')
if img_phone is None:
    print("无法读取phone.jpg图像")
    exit()

# 缩放phone.jpg至与all.jpg一致
resized_img_phone = cv2.resize(img_phone, (width_all, height_all))

# 保存缩放后的图像为phone_resized.jpg
cv2.imwrite('phone_resized.jpg', resized_img_phone)
print("已成功缩放phone.jpg并保存为phone_resized.jpg！")

# 不需要调用 cap.release() 和 cv2.destroyAllWindows()，可以删去这两行
