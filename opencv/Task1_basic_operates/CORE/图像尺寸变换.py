import cv2

img = cv2.imread('phone.jpg', flags=cv2.IMREAD_COLOR)

size = (640, 480)
myImg = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

# 使用cv2.imwrite保存图片，第一个参数为保存的文件名，第二个参数为要保存的图像数据
cv2.imwrite("phone_resized.jpg", myImg)

cv2.namedWindow("my resize picture")
cv2.imshow("my resize picture", myImg)

cv2.waitKey(0)
cv2.destroyAllWindows()