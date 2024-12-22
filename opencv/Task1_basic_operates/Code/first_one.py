import cv2

ca = cv2.VideoCapture(0)

print("按 “s” 键拍照保存，按 “q” 退出程序")

while True:
    ret, fr = ca.read()

    cv2.imshow('Camera', fr)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite('all.jpg', fr)
        print("照片已保存为 all.jpg")
    elif key == ord('q'):
        break

ca.release()
cv2.destroyAllWindows()