import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 100, 200)

    kernel = np.ones((5, 5), np.uint8)  
    dilated = cv2.dilate(edges, kernel, iterations=1)  
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        center_x, center_y = x + w // 2, y + h // 2
        
        white_color = (255, 255, 255)

        cv2.drawContours(frame, [contour], -1, white_color, 2)

        cv2.drawMarker(frame, (center_x, center_y), white_color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

