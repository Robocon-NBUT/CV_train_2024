#include <opencv2/opencv.hpp>
#include <iostream>

int main() {

    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "�޷�������ͷ��" << std::endl;
        return -1;
    }

    cv::Mat frame;

    cap >> frame;

    if (frame.empty()) {
        std::cerr << "�޷�����ͼ��" << std::endl;
        return -1;
    }

    bool isSaved = cv::imwrite("all.jpg", frame);

    if (isSaved) {
        std::cout << "ͼ���ѱ���Ϊ all.jpg" << std::endl;
    }
    else {
        std::cerr << "����ͼ��ʧ�ܣ�" << std::endl;
    }

    cap.release();

    return 0;
}