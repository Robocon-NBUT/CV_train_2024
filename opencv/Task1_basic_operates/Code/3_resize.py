import cv2


def resize_phone_to_all():

    # 读取 all.jpg
    all_image = cv2.imread("all.jpg")
    if all_image is None:
        print(f"无法打开图片")
        return

    # 获取 all.jpg 的尺寸
    all_height, all_width = all_image.shape[:2]
    print(f"all.jpg 的尺寸: 宽度={all_width}, 高度={all_height}")

    # 读取 phone.jpg
    phone_image = cv2.imread("phone.jpg")
    if phone_image is None:
        print(f"无法打开图片")
        return

    # 获取 phone.jpg 的原始尺寸
    phone_height, phone_width = phone_image.shape[:2]
    print(f"phone.jpg 的原始尺寸: 宽度={phone_width}, 高度={phone_height}")

    # 调整 phone.jpg 的尺寸为 all.jpg 的尺寸
    resized_phone = cv2.resize(phone_image, (all_width, all_height), interpolation=cv2.INTER_AREA)

    # 保存调整后的图像
    cv2.imwrite("phone_resized.jpg", resized_phone)
    print(f"调整后的图片已保存")



resize_phone_to_all()
