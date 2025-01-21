#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main()
{
    // 读取 all.jpg
    cv::Mat image = cv::imread("all.jpg");
    if (image.empty())
        return -1;

    // 转换为灰度图并保存
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    cv::imwrite("all_gray.jpg", gray_image);

    // 转换为HSV色彩空间并保存
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
    cv::imwrite("all_hsv.jpg", hsv_image);

    // 转换为LAB色彩空间并保存
    cv::Mat lab_image;
    cv::cvtColor(image, lab_image, cv::COLOR_BGR2Lab);
    cv::imwrite("all_lab.jpg", lab_image);

    return 0;
}
