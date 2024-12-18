import cv2 as cv
import numpy as np
def imshow(name,image):
    cv.imshow(name,image)
    cv.waitKey(0)
    cv.destroyAllWindows()
phone=cv.imread(r'C:\Users\20936\Desktop\TASK\all.jpg')
phone1=phone[89:297,120:590]
imshow("shouji",phone1)
if cv.imwrite(r'C:\Users\20936\Desktop\TASK\phone.jpg',phone1):
    print("保存成功")
else:
    print("保存失败")
