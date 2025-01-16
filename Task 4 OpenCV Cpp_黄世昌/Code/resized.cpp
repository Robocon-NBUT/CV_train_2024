#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat phone = cv::imread("phone.jpg");
    if (phone.empty()) {
        std::cerr << "Error: Could not load phone image." << std::endl;
        return -1;
    }

    cv::Mat all = cv::imread("all.jpg");
    if (all.empty()) {
        std::cerr << "Error: Could not load all image." << std::endl;
        return -1;
    }

    cv::Mat resized_phone;
    cv::resize(phone, resized_phone, all.size());

    cv::imwrite("phone_resized.jpg", resized_phone);
    std::cout << "Resized phone image saved as phone_resized.jpg" << std::endl;

    return 0;
}