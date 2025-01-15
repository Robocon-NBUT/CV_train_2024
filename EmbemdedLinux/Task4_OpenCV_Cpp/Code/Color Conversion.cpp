#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
   
    cv::Mat img = cv::imread("D:/opencv_test/all.jpg");
    if (img.empty()) {
        std::cout << "Unable to read picture" << std::endl;
        return -1;
    }

    
    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    if (!cv::imwrite("all_gray.jpg", gray_img)) {
        std::cout << " Unable to save grey release" << std::endl;
        return -1;
    }

   
    cv::Mat hsv_img;
    cv::cvtColor(img, hsv_img, cv::COLOR_BGR2HSV);
    if (!cv::imwrite("all_hsv.jpg", hsv_img)) {
        std::cout << "Unable to save HSV release" << std::endl;
        return -1;
    }


    cv::Mat lab_img;
    cv::cvtColor(img, lab_img, cv::COLOR_BGR2Lab);
    if (!cv::imwrite("all_lab.jpg", lab_img)) {
        std::cout << "Unable to save Lab release" << std::endl;
        return -1;
    }

    std::cout << "The picture is saved." << std::endl;
    return 0;
}