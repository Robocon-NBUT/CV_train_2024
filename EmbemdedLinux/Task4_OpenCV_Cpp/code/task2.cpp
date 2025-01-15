#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat allImage = cv::imread("all.jpg");


    cv::Mat phoneImage = cv::imread("phone.jpg");



    int allWidth = allImage.cols;
    int allHeight = allImage.rows;


    cv::Mat phoneResized;
    cv::resize(phoneImage, phoneResized, cv::Size(allWidth, allHeight));


    cv::imwrite("phone_resized.jpg", phoneResized);


    cv::imshow("Phone_resized Image", phoneResized);
    cv::waitKey(0);

    std::cout << "successful" << std::endl;

    return 0;
}