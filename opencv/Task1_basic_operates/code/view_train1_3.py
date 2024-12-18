import cv2 as cv
import numpy as np
def imshow(name,image):
    cv.imshow(name,image)
    cv.waitKey(0)
    cv.destroyAllWindows()
environment=cv.imread(r'C:\Users\20936\Desktop\TASK\all.jpg')
phone=cv.imread(r'C:\Users\20936\Desktop\TASK\phone.jpg')
height,width,_=environment.shape
bigphone=cv.resize(phone,(width,height))
if cv.imwrite(r'C:\Users\20936\Desktop\TASK\phone_resized.jpg',bigphone):
    print("保存成功")
else:
    print("保存失败")
imshow("shouji",bigphone)
imshow("huanjing",environment)