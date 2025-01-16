#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 读取原始图片
    cv::Mat phone = cv::imread("phone.jpg");
    cv::Mat all = cv::imread("all.jpg");

    // 检查图片是否读取成功
    if (phone.empty()) {
        std::cout << "无法打开或找到 phone.jpg" << std::endl;
        return -1;
    }
    if (all.empty()) {
        std::cout << "无法打开或找到 all.jpg" << std::endl;
        return -1;
    }

    // 获取 all.jpg 的尺寸
    cv::Size allSize = all.size();

    // 缩放 phone.jpg 使其尺寸与 all.jpg 一致
    cv::Mat phone_resized;
    cv::resize(phone, phone_resized, allSize);

    // 保存处理后的图片
    cv::imwrite("phone_resized.jpg", phone_resized);

    std::cout << "图片已保存为 phone_resized.jpg" << std::endl;

    return 0;
}
