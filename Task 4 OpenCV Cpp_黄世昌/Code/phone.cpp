#include <opencv2/opencv.hpp>
#include <iostream>

int main() {

    cv::Mat image = cv::imread("D:\\»ÆÊÀ²ý\\Desktop\\Task_4\\Task_4\\all.jpg");


    if (image.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    cv::Rect roi(50, 100, 500, 300);
    cv::Mat phone = image(roi);

    cv::imwrite("phone.jpg", phone);
    std::cout << "Phone image saved as phone.jpg" << std::endl;

    return 0;
}