#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    // 读取 all.jpg 图像
    cv::Mat image = cv::imread("/root/Desktop/ImageShow/my_venv/photograph/all.jpg");
    if (image.empty()) {
        std::cerr << "Could not open or find the image: all.jpg" << std::endl;
        return -1;
    }

    // 假设手机部分的坐标和尺寸（根据实际情况修改）
    int x = 5;  // 截取区域左上角 x 坐标
    int y = 105;  // 截取区域左上角 y 坐标
    int width = 616;  // 截取区域的宽度
    int height = 290; // 截取区域的高度
    cv::Rect roi(x, y, width, height);

    // 截取手机部分
    cv::Mat phone = image(roi);

    // 保存截取的图像为 phone.jpg
    if (!cv::imwrite("phone.jpg", phone)) {
        std::cerr << "Could not save the image: phone.jpg" << std::endl;
        return -1;
    }

    std::cout << "Image saved successfully as phone.jpg" << std::endl;
    return 0;
}
