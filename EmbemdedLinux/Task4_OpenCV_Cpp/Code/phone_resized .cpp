#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img = cv::imread("/opencv.cpp/phone.jpg");
    if (img.empty()) {
        std::cout << "Unable to read picture " << std::endl;
        return -1;
    }

    cv::Point2f center(img.cols / 2.0f, img.rows / 2.0f);
    cv::Mat m2 = cv::getRotationMatrix2D(center, 90, 1);
    cv::Mat The_flipped_image;
    cv::warpAffine(img, The_flipped_image, m2, img.size());

    int target_width = 640;
    int target_height = 480;

    cv::Mat resized_img;
    cv::resize(The_flipped_image, resized_img, cv::Size(target_width, target_height));


    if (!cv::imwrite("phone_resized.jpg", resized_img)) {
        std::cout << "Unable to save picture" << std::endl;
        return -1;
    }
    return 0;
}