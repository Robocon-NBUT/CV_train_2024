import cv2

def caijian(lujing):

    img = cv2.imread(lujing, cv2.IMREAD_GRAYSCALE)
    Output = img[110:300, 90:490]
    cv2.imshow('op', Output)
    cv2.waitKey(0)
    cv2.imwrite('phone.jpg',Output)
    cv2.destroyAllWindows()
    return Output

Output=caijian('all.jpg')