#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    // 读取 all.jpg 图像
    cv::Mat image = cv::imread("/root/Desktop/ImageShow/my_venv/photograph/all.jpg");
    if (image.empty()) {
        std::cerr << "无法读取 /root/Desktop/ImageShow/my_venv/photograph/all.jpg 文件，请检查文件是否存在及路径是否正确" << std::endl;
        return -1;
    }

    // 转换为灰度颜色空间
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    if (!cv::imwrite("all_gray.jpg", gray_image)) {
        std::cerr << "无法保存灰度颜色空间图像为 all_gray.jpg" << std::endl;
        return -1;
    }
    std::cout << "已成功保存灰度颜色空间图像为 all_gray.jpg" << std::endl;

    // 转换为 HSV 颜色空间
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
    if (!cv::imwrite("all_hsv.jpg", hsv_image)) {
        std::cerr << "无法保存 HSV 颜色空间图像为 all_hsv.jpg" << std::endl;
        return -1;
    }
    std::cout << "已成功保存 HSV 颜色空间图像为 all_hsv.jpg" << std::endl;

    // 转换为 LAB 颜色空间
    cv::Mat lab_image;
    cv::cvtColor(image, lab_image, cv::COLOR_BGR2Lab);
    if (!cv::imwrite("all_lab.jpg", lab_image)) {
        std::cerr << "无法保存 LAB 颜色空间图像为 all_lab.jpg" << std::endl;
        return -1;
    }
    std::cout << "已成功保存 LAB 颜色空间图像为 all_lab.jpg" << std::endl;

    return 0;
}
