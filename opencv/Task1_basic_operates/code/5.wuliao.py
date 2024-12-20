import cv2
import numpy as np


# 在摄像头中用相反的颜色标出给定的物料轮廓，并用与轮廓相同的颜色用十字标出物料外接矩形的几何中心

def detect1():

    cv2.namedWindow('came', 0)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        cv2.imshow('came', frame)

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_frame, lower_red, upper_red)
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = []

        min_area_threshold=600

        #max_area_threshold=400

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > min_area_threshold :  #and area<max_area_threshold
                filtered_contours.append(contour)

        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)

            contour_color = frame[y, x]




            if isinstance(contour_color, np.ndarray):
                if contour_color.ndim > 1:
                    contour_color = tuple(contour_color[0])
                else:
                    contour_color = tuple(int(c) for c in contour_color)
            elif isinstance(contour_color, list):
                contour_color = tuple(contour_color)
            else:
                raise ValueError("contour_color has an unexpected data type")


            opposite_color = tuple(255 - int(c) for c in contour_color)

            cv2.drawContours(frame, [contour], -1, opposite_color, 2)

            center_x = x + w // 2
            center_y = y + h // 2

            cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), contour_color, 2)
            cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), contour_color, 2)

            cv2.rectangle(frame, (x, y), (x + w, y+ h), contour_color, 2)

            cv2.imshow('came', frame)

        key = cv2.waitKey(10)

        if key == 32:
            break

    cap.release()
    cv2.destroyAllWindows()

detect1()