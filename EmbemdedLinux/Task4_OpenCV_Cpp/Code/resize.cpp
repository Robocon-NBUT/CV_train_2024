#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

void resize_phone_to_all()
{
    // 读取 all.jpg
    cv::Mat all_image = cv::imread("all.jpg");
    if (all_image.empty())
    {
        std::cerr << "无法打开 all.jpg" << std::endl;
        return;
    }

    // 获取 all.jpg 的尺寸
    int all_height = all_image.rows;
    int all_width = all_image.cols;
    std::cout << "all.jpg 的尺寸: 宽度 = " << all_width << ", 高度 = " << all_height << std::endl;

    // 读取 phone.jpg
    cv::Mat phone_image = cv::imread("phone.jpg");
    if (phone_image.empty())
    {
        std::cerr << "无法打开 phone.jpg" << std::endl;
        return;
    }

    // 获取 phone.jpg 的原始尺寸
    int phone_height = phone_image.rows;
    int phone_width = phone_image.cols;
    std::cout << "phone.jpg 的原始尺寸: 宽度 = " << phone_width << ", 高度 = " << phone_height << std::endl;

    // 将 phone.jpg 调整大小以匹配 all.jpg 的尺寸
    cv::Mat resized_phone;
    cv::resize(phone_image, resized_phone, cv::Size(all_width, all_height), 0, 0, cv::INTER_AREA);

    // 保存调整大小后的图像
    if (cv::imwrite("phone_resized.jpg", resized_phone))
    {
        std::cout << "调整大小后的图像已保存为 phone_resized.jpg" << std::endl;
    }
    else
    {
        std::cerr << "保存调整大小后的图像失败" << std::endl;
    }
}

int main()
{
    resize_phone_to_all();
    return 0;
}
