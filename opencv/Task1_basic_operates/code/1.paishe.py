import cv2

path ='D:\\1\\opencv'

cv2.namedWindow('came', 0)
cap = cv2.VideoCapture(0)



ret, frame = cap.read()
cv2.imshow(frame)
key = cv2.waitKey(10)
if key == 32:
    cv2.imwrite(path + '/all.jpg', frame)
    cap.release()
    