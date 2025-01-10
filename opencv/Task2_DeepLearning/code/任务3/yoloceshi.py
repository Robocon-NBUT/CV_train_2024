from ultralytics import YOLO
import cv2

model = YOLO('D:\\1\\yolo\\runs\\train\\exp\\weights\\best.pt')

photo = cv2.VideoCapture(0)


category_mapping = {
    1: 'huishou',
    2: 'huishou',
    5: 'huishou',
    6: 'huishou',
    7: 'huishou',
    8: 'huishou',
    0: 'youhai',
    3: 'youhai',
    4: 'chuyu'
}

while True:
    ret, frame = photo.read()
    if not ret:
        break
    results = model.predict(frame)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            cls_index = int(box.cls[0])
            clsname = model.names[cls_index]
            category =  category_mapping[cls_index]


            print('category_mapping:', category)
            print('model.names:', clsname)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{clsname}: {category}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0))
    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == 32:
        break
photo.release()
cv2.destroyAllWindows()






            


