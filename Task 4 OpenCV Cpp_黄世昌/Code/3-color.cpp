#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat image = cv::imread("all.jpg");
    if (image.empty()) 
    {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }


    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::imwrite("all_gray.jpg", gray);


    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    cv::imwrite("all_hsv.jpg", hsv);


    cv::Mat lab;
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
    cv::imwrite("all_lab.jpg", lab);

    std::cout << "Images saved as all_gray.jpg, all_hsv.jpg, all_lab.jpg" << std::endl;

    return 0;
}