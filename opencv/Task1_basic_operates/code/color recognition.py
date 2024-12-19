import cv2
import numpy as np
 
 
lower_red = np.array([0, 50, 100])
upper_red = np.array([10, 255, 255])
lower_blue = np.array([100,50,100])
upper_blue = np.array([124, 255, 255])     #若绘制轮廓与自己期望的识别结果相差较大，可通过调整阈值来改变识别结果
red = (0,0,225)
blue = (225,0,0)
 
 
cv2.namedWindow('video', cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow('video',640,480)
 
def img_process(img,lower,upper):
    """根据阈值处理图像，提取阈值内的颜色。返回处理后只留下指定颜色的图像（其余为黑色）
        img：原图像；lower：最低阈值；upper：最高阈值"""
    kernel = np.ones((35, 35), np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Open = cv2.morphologyEx(hsv,cv2.MORPH_OPEN,kernel)
    mask = cv2.inRange(Open, lower, upper)
    res = cv2.bitwise_and(img, img, mask = mask)
    return res
 
def cnts_draw(img,res,color):
    """在原图像上绘出指定颜色的轮廓。无返回值
        img：原图像；res：只剩某颜色的位与运算后的图像；color：轮廓的颜色"""
    canny = cv2.Canny(res,100,200)
    contours, hierarchy=cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        cv2.imshow('video',img)
        return
    else:
        max_cnt = max(contours , key = cv2.contourArea)
        cv2.drawContours(img, max_cnt,-1,color,2)
        (x,y,w,h) = cv2.boundingRect(max_cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),color,3)
        cv2.imshow('video', img)
 
        
def colorfind(img):
    """找到原图像最多的颜色，当该颜色为红色或蓝色时打印出来该颜色的名称，无返回值
        img：原图像"""
    kernel = np.ones((35, 35), np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Open = cv2.morphologyEx(hsv,cv2.MORPH_OPEN,kernel)
    hist = cv2.calcHist([Open],[0],None,[180],[0,180])
    hist_max = np.where(hist == np.max(hist))
    if 0 < hist_max[0] < 10:
        print('red')
    elif 100 < hist_max[0] < 124:
        print('blue')
    else :
        return
 
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            print("无法读取摄像头！")
            break
        else:
            if frame is not None: 
                res_blue = img_process(frame,lower_blue,upper_blue)
                res_red = img_process(frame,lower_red,upper_red)
                cnts_draw(frame,res_blue,blue)
                cnts_draw(frame,res_red,red)
                colorfind(frame)
                key = cv2.waitKey(0)
                if key == 27:
                    break
            else:
                print("无画面")
                break
 
 
    cap.release()
    cv2.destroyAllWindows()