#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img = cv::imread("/opencv.cpp/all.jpg");
    if (img.empty()) {
        std::cout << "Unable to read picture" << std::endl;
        return -1;
    }

    int x1 = 311, y1 = 92, x2 = 478, y2 = 440;
    cv::Mat phonePart = img(cv::Rect(x1, y1, x2 - x1, y2 - y1));

    if (!cv::imwrite("phone.jpg", phonePart)) {
        std::cout << "Unable to save picture" << std::endl;
        return -1;
    }

    std::cout << "Image saved successfully" << std::endl;
    return 0;
}