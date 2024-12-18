import  cv2 as cv
def imshow(name,image):
    cv.imshow(name,image)
    cv.waitKey(0)
    cv.destroyAllWindows()
phone=cv.VideoCapture(0)
while 1:
    ret,phone1=phone.read()
    if ret:
        cv.imshow("all", phone1)
        key=cv.waitKey(25)
        if key==32:
            if cv.imwrite(r'C:\Users\20936\Desktop\TASK\all.jpg',phone1):
                print("成功保存")
            else:
                print("保存失败")
        elif key==113:
            break
    else:
        break
phone.release()