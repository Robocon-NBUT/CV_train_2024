#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    // 读取 all.jpg 图像
    cv::Mat all_image = cv::imread("/root/Desktop/ImageShow/my_venv/photograph/all.jpg");
    if (all_image.empty()) {
        std::cerr << "无法读取 all.jpg 文件，请检查文件是否存在及路径是否正确" << std::endl;
        return -1;
    }
    int height_all = all_image.rows;
    int width_all = all_image.cols;

    // 读取 phone.jpg 图像
    cv::Mat phone_image = cv::imread("/root/Desktop/ImageShow/my_venv/phone.jpg");
    if (phone_image.empty()) {
        std::cerr << "无法读取 phone.jpg 文件，请检查文件是否存在及路径是否正确" << std::endl;
        return -1;
    }
    int height_phone = phone_image.rows;
    int width_phone = phone_image.cols;

    // 计算缩放比例
    double scale_x = static_cast<double>(width_all) / width_phone;
    double scale_y = static_cast<double>(height_all) / height_phone;

    // 根据缩放比例对 phone.jpg 图像进行缩放
    cv::Mat resized_phone_image;
    cv::resize(phone_image, resized_phone_image, cv::Size(0, 0), scale_x, scale_y);

    // 保存缩放后的图像为 phone_resized.jpg
    if (!cv::imwrite("phone_resized.jpg", resized_phone_image)) {
        std::cerr << "无法保存 phone_resized.jpg 文件" << std::endl;
        return -1;
    }
    std::cout << "phone.jpg 已成功缩放并保存为 phone_resized.jpg" << std::endl;
    return 0;
}
