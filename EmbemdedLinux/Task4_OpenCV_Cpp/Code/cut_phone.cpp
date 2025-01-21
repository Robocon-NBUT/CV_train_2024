#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        std::cerr << "用法: " << argv[0] << " <x> <y> <宽度> <高度>" << std::endl;
        return -1;
    }

    // 解析命令行参数
    int x = std::stoi(argv[1]);
    int y = std::stoi(argv[2]);
    int w = std::stoi(argv[3]);
    int h = std::stoi(argv[4]);

    // 读取图像
    cv::Mat image = cv::imread("all.jpg");

    if (image.empty())
    {
        std::cerr << "无法打开图像" << std::endl;
        return -1;
    }

    // 验证感兴趣区域 (ROI) 的有效性
    if (w == 0 || h == 0)
    {
        std::cout << "区域大小无效，退出裁剪。" << std::endl;
        return 0;
    }

    // 确保ROI在图像边界内
    if (x < 0 || y < 0 || x + w > image.cols || y + h > image.rows)
    {
        std::cerr << "ROI 超出图像边界" << std::endl;
        return -1;
    }

    // 裁剪图像
    cv::Rect roi(x, y, w, h);
    cv::Mat cropped_image = image(roi).clone();

    // 保存裁剪后的图像
    if (cv::imwrite("phone.jpg", cropped_image))
    {
        std::cout << "裁剪后的图像已保存为 phone.jpg" << std::endl;
    }
    else
    {
        std::cerr << "保存图像失败" << std::endl;
        return -1;
    }

    return 0;
}
