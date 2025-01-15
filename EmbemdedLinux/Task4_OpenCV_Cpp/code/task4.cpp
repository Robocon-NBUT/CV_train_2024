#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {

    cv::Mat frame = cv::imread("task4/task4.jpg");

    cv::Mat hsv_frame;
    cv::cvtColor(frame, hsv_frame, cv::COLOR_BGR2HSV);

    cv::Mat mask1, mask2, mask;
    cv::Scalar lower_red1(0, 100, 100);
    cv::Scalar upper_red1(10, 255, 255);
    cv::Scalar lower_red2(160, 100, 100);
    cv::Scalar upper_red2(179, 255, 255);

    cv::inRange(hsv_frame, lower_red2, upper_red2, mask2);
    cv::bitwise_or(mask1, mask2, mask);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> filtered_contours;
    int min_area_threshold = 600;

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > min_area_threshold) {
            filtered_contours.push_back(contour);
        }
    }

    for (const auto& contour : filtered_contours) {
        cv::Rect boundingRect = cv::boundingRect(contour);
        int x = boundingRect.x;
        int y = boundingRect.y;
        int w = boundingRect.width;
        int h = boundingRect.height;
        cv::Vec3b contour_color = frame.at<cv::Vec3b>(y, x);
        cv::Scalar opposite_color(255 - contour_color[0], 255 - contour_color[1], 255 - contour_color[2]);

        cv::drawContours(frame, std::vector<std::vector<cv::Point>>{contour}, -1, opposite_color, 2);

        int center_x = x + w / 2;
        int center_y = y + h / 2;

        cv::line(frame, cv::Point(center_x, center_y - 10), cv::Point(center_x, center_y + 10), cv::Scalar(contour_color[0], contour_color[1], contour_color[2]), 2);
        cv::line(frame, cv::Point(center_x - 10, center_y), cv::Point(center_x + 10, center_y), cv::Scalar(contour_color[0], contour_color[1], contour_color[2]), 2);
        cv::rectangle(frame, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(contour_color[0], contour_color[1], contour_color[2]), 2);
    }

    cv::imshow("came", frame);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}