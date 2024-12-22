import cv2
import numpy as np

def ExOrd_Eyes():

    cv2.namedWindow('window', 0)
    op = cv2.VideoCapture(0)

    while 1:
        ret, fr = op.read()
        cv2.imshow('window', fr)

        hsv_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        k1 = cv2.inRange(hsv_fr, lower_red, upper_red)
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        s2 = cv2.inRange(hsv_fr, lower_red2, upper_red2)
        k = cv2.bitwise_or(k1, s2)
        contours, chy = cv2.findContours(k, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        min_area_threshold=600

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area_threshold:
                filtered_contours.append(contour)

        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            contour_color = fr[y, x]
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
            cv2.drawContours(fr, [contour], -1, opposite_color, 2)
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.line(fr, (center_x, center_y - 10), (center_x, center_y + 10), contour_color, 2)
            cv2.line(fr, (center_x - 10, center_y), (center_x + 10, center_y), contour_color, 2)
            cv2.rectangle(fr, (x, y), (x + w, y+ h), contour_color, 2)
            cv2.imshow('window', fr)

        btn = cv2.waitKey(10)
        if btn == ord('s'):
            cv2.imwrite('wu_liao.jpg', fr)
            print("照片已保存为 wu_liao.jpg")
        elif btn == ord('q'):
            break

    op.release()
    cv2.destroyAllWindows()

ExOrd_Eyes()