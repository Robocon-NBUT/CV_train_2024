import cv2
cap = cv2.VideoCapture(0)

while(True):
    ret,frame = cap.read()
    if ret is None:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1=(0,150,100)
    mask2=(10,255,255)
    mask3=(170,150,100)
    mask4=(180,255,255)

    erzhi1 = cv2.inRange(hsv,mask1,mask2)
    erzhi2 = cv2.inRange(hsv,mask3,mask4)
    erzhi = cv2.bitwise_or(erzhi1,erzhi2)

    erzhi = cv2.medianBlur(erzhi,5)
    con=cv2.findContours(erzhi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    for contour in con:
        area= cv2.contourArea(contour)
        if area > 1000:
            x,y,w,h = cv2.boundingRect(contour)
            yanse = frame[y:y+h,x:x+w]
            yan = cv2.mean(yanse)[:3]
            fyanse = (255-yan[0],255-yan[1],255-yan[2])

            cv2.drawContours(frame,[contour],-1,fyanse,2)
            tx = x+w//2
            ty = y+h//2

            cv2.drawMarker(frame,[tx,ty],fyanse,cv2.MARKER_CROSS,10,2)
    cv2.imshow('frame',frame)
    cv2.imshow('erzhi',erzhi)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()