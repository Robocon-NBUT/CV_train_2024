#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 读取 all.jpg 图片
    cv::Mat all = cv::imread("all.jpg");

    // 检查图片是否成功读取
    if (all.empty()) {
        std::cout << "无法打开或找到 all.jpg" << std::endl;
        return -1;
    }

    // 转换为灰度图
    cv::Mat all_gray;
    cv::cvtColor(all, all_gray, cv::COLOR_BGR2GRAY);
    // 保存灰度图
    cv::imwrite("all_gray.jpg", all_gray);

    // 转换为 HSV 颜色空间
    cv::Mat all_hsv;
    cv::cvtColor(all, all_hsv, cv::COLOR_BGR2HSV);
    // 保存 HSV 图
    cv::imwrite("all_hsv.jpg", all_hsv);

    // 转换为 LAB 颜色空间
    cv::Mat all_lab;
    cv::cvtColor(all, all_lab, cv::COLOR_BGR2Lab);
    // 保存 LAB 图
    cv::imwrite("all_lab.jpg", all_lab);

    std::cout << "图片已保存为 all_gray.jpg, all_hsv.jpg, all_lab.jpg" << std::endl;

    return 0;
}
