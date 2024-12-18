import cv2 as cv
import numpy as np
def imshow(name,image):
    cv.imshow(name,image)
    cv.waitKey(0)
    cv.destroyAllWindows()
def save(name,image):
    file_name=f'C:/Users/20936/Desktop/TASK/{name}.jpg'
    if cv.imwrite(file_name,image):
        print("保存成功")
environment=cv.imread(r'C:\Users\20936\Desktop\TASK\all.jpg')
environment_gray=cv.cvtColor(environment,cv.COLOR_BGR2GRAY)
save('all_gray',environment_gray)
environment_hsv=cv.cvtColor(environment,cv.COLOR_BGR2HSV)
save('all_hsv',environment_hsv)
environment_lab=cv.cvtColor(environment,cv.COLOR_BGR2Lab)
save('all_lab',environment_lab)
