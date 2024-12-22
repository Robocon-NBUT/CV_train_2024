import cv2
import numpy as np
#调用笔记本内置摄像头，所以参数为0，如果有其他的摄像头可以调整参数为1，2
cap = cv2.VideoCapture(0)
while True:
    #从摄像头读取图片
    sucess, img = cap.read()
    #转为灰度图片（这一行可以注释掉，因为现在不需要灰度图了）
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #显示摄像头，背景是彩色。
    cv2.imshow("img", img)
    #保持画面的持续。
    k = cv2.waitKey(1)
    if k == 27:
        #通过esc键退出摄像
        cv2.destroyAllWindows()
        break
    elif k == ord("s"):
        #通过s键保存图片，并退出。
        cv2.imwrite("image2.jpg", img)
        cv2.destroyAllWindows()
        break
#关闭摄像头
cap.release()