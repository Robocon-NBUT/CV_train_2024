import cv2

img = cv2.imread('all.jpg')

phone_img = img[200:430, 50:500]

cv2.imwrite('phone.jpg', phone_img)
print("手机部分已保存为 phone.jpg")
