#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 定义输入和输出图像路径
    std::string allImagePath = "all.jpg";  // 输入图片路径
    std::string phoneImagePath = "phone.jpg";  // 截取后的手机图片路径
    std::string phoneResizedPath = "phone_resized.jpg";  // 缩放后的手机图片路径

    // 加载 all.jpg 图像
    cv::Mat allImg = cv::imread(allImagePath);
    if (allImg.empty()) {
        std::cerr << "加载 all.jpg 图像失败，请检查路径: " << allImagePath << std::endl;
        return -1;
    }
    std::cout << "成功加载 all.jpg 图像" << std::endl;

    // 1. 根据给定坐标截取手机部分并保存为 phone.jpg
    int x = 290;       // 左上角 x 坐标
    int y = 94;        // 左上角 y 坐标
    int width = 209;   // 截取区域宽度
    int height = 355;  // 截取区域高度

    // 检查截取区域是否超出图像范围
    if (x < 0 || y < 0 || x + width > allImg.cols || y + height > allImg.rows) {
        std::cerr << "错误: 截取区域超出图像范围！" << std::endl;
        return -1;
    }

    // 截取手机部分
    cv::Rect roi(x, y, width, height);
    cv::Mat phoneImg = allImg(roi);

    // 保存截取的手机部分为 phone.jpg
    if (!cv::imwrite(phoneImagePath, phoneImg)) {
        std::cerr << "保存 phone.jpg 失败！" << std::endl;
        return -1;
    }
    std::cout << "截取的手机图像已保存为: " << phoneImagePath << std::endl;

    // 2. 将 phone.jpg 缩放至与 all.jpg 一致，并保存为 phone_resized.jpg
    cv::Mat resizedPhoneImg;
    cv::resize(phoneImg, resizedPhoneImg, allImg.size());  // 缩放尺寸与 all.jpg 一致

    // 保存缩放后的图片
    if (!cv::imwrite(phoneResizedPath, resizedPhoneImg)) {
        std::cerr << "保存 phone_resized.jpg 失败！" << std::endl;
        return -1;
    }
    std::cout << "缩放后的手机图像已保存为: " << phoneResizedPath << std::endl;

    // 3. 将 all.jpg 转换为灰度、HSV、Lab 三种颜色空间并分别保存
    cv::Mat grayImg, hsvImg, labImg;

    // 转换为灰度图像
    cv::cvtColor(allImg, grayImg, cv::COLOR_BGR2GRAY);
    if (!cv::imwrite("all_gray.jpg", grayImg)) {
        std::cerr << "保存 all_gray.jpg 失败！" << std::endl;
        return -1;
    }
    std::cout << "灰度图像已保存为: all_gray.jpg" << std::endl;

    // 转换为 HSV 图像
    cv::cvtColor(allImg, hsvImg, cv::COLOR_BGR2HSV);
    if (!cv::imwrite("all_hsv.jpg", hsvImg)) {
        std::cerr << "保存 all_hsv.jpg 失败！" << std::endl;
        return -1;
    }
    std::cout << "HSV 图像已保存为: all_hsv.jpg" << std::endl;

    // 转换为 Lab 图像
    cv::cvtColor(allImg, labImg, cv::COLOR_BGR2Lab);
    if (!cv::imwrite("all_lab.jpg", labImg)) {
        std::cerr << "保存 all_lab.jpg 失败！" << std::endl;
        return -1;
    }
    std::cout << "Lab 图像已保存为: all_lab.jpg" << std::endl;

    // 显示结果图像
    cv::imshow("Original Image", allImg);
    cv::imshow("Cropped Phone Image", phoneImg);
    cv::imshow("Resized Phone Image", resizedPhoneImg);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
