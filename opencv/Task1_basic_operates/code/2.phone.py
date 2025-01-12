import cv2




def chuli(img_path):    #打开all.jpg并截取手机部分，保存图片为phone.jpg

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img1 = img[220:430, 90:530]
    cv2.imshow('666', img1)
    cv2.waitKey(0)
    cv2.imwrite('phone.jpg',img1)
    cv2.destroyAllWindows()
    return img1

img1=chuli('all.jpg')
