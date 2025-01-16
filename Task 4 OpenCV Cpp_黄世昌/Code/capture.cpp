#include <opencv2/opencv.hpp>
#include <iostream>

int main() {

    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头！" << std::endl;
        return -1;
    }

    cv::Mat frame;

    cap >> frame;

    if (frame.empty()) {
        std::cerr << "无法捕获图像！" << std::endl;
        return -1;
    }

    bool isSaved = cv::imwrite("all.jpg", frame);

    if (isSaved) {
        std::cout << "图像已保存为 all.jpg" << std::endl;
    }
    else {
        std::cerr << "保存图像失败！" << std::endl;
    }

    cap.release();

    return 0;
}