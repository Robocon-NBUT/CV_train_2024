import cv2 as cv
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
model=YOLO("datas/detect/runs/train/num1/weights/best.pt")
photo=cv.VideoCapture(0)
rubbish_sort={'pen':'other','stone':'other','brick':'other','bottle':'recyclable','waterbottle':'recyclable','cup':'recyclable','battery':'hazardous','drug':'hazardous','modelbattery':'hazardous'}
def cv2ImgAddText(img, text, left, top, textColor=(255, 0, 0), textSize=20):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        fontText = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", textSize, encoding="utf-8")
        draw.text((left, top), text, textColor, font=fontText)
        return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
while True:
    ret,frame=photo.read()
    if not ret:
        break
    results=model.predict(frame)
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=[int(i) for i in box.xyxy[0]]
            cls_index=int(box.cls[0])
            clsname=model.names
            clsname=clsname[cls_index]
            if clsname in rubbish_sort:
                clsname_chinese=rubbish_sort[clsname]
            else:
                clsname_chinese=clsname
            frame=cv2ImgAddText(frame,clsname,0,10,textColor=(255,0,0))
            cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv.putText(frame,clsname_chinese,(x1,y1-10),cv.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0))
    cv.imshow('video',frame)
    if cv.waitKey(1)&0xFF==32:
        break
photo.release()
cv.destroyAllWindows()