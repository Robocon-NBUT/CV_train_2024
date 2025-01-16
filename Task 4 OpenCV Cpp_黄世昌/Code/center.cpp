#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam." << std::endl;
        return -1;
    }

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Could not capture frame." << std::endl;
            break;
        }


        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);


        cv::Scalar lower_orange1 = cv::Scalar(10, 100, 100);
        cv::Scalar upper_orange1 = cv::Scalar(20, 255, 255);
        cv::Scalar lower_orange2 = cv::Scalar(20, 100, 100);
        cv::Scalar upper_orange2 = cv::Scalar(30, 255, 255);

        cv::Mat mask1, mask2, mask;
        cv::inRange(hsv, lower_orange1, upper_orange1, mask1);
        cv::inRange(hsv, lower_orange2, upper_orange2, mask2);
        cv::bitwise_or(mask1, mask2, mask);


        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1);


        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            if (cv::contourArea(contour) < 50) {
                continue;
            }

            cv::Rect rect = cv::boundingRect(contour);
            int x = rect.x;
            int y = rect.y;
            int w = rect.width;
            int h = rect.height;

            int center_x = x + w / 2;
            int center_y = y + h / 2;

            cv::Scalar white_color(255, 255, 255);

            cv::drawContours(frame, std::vector<std::vector<cv::Point>>{contour}, -1, white_color, 2);

            cv::drawMarker(frame, cv::Point(center_x, center_y), white_color, cv::MARKER_CROSS, 10, 2);
        }

        cv::imshow("Webcam", frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}