import cv2


def cut_phone():
    # 读取图片
    image = cv2.imread("all.jpg")

    if image is None:
        print(f"无法打开图片")
        return

    # 显示图片并让用户选择裁剪区域
    roi = cv2.selectROI("选择裁剪区域", image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    # 解包 ROI 坐标
    x, y, w, h = roi

    if w == 0 or h == 0:
        print("未选择任何区域，退出裁剪。")
        return

    # 裁剪图片
    cropped_image = image[y:y + h, x:x + w]

    # 保存裁剪后的图片
    cv2.imwrite('phone.jpg',cropped_image)
    print(f"裁剪后的图片已保存")

cut_phone()