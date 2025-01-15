import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("could not open camera")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
while True:
    ret,frame=cap.read()
    if not ret:
        print("failed")
        break
    cv2.imshow('photo',frame )
    key=cv2.waitKey(1)&0xFF
    if key==ord('s'):
        cv2.imwrite("all.jpg",frame)
        print("Image save as 'all.ipg'.")
        cap.release()