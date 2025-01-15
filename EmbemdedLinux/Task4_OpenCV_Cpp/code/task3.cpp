#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat allImage = cv::imread("all.jpg");

    cv::Mat allGray;
    cv::cvtColor(allImage, allGray, cv::COLOR_BGR2GRAY);
    cv::imwrite("all_gray.jpg", allGray);

    cv::imshow("All_gray", allGray);
    cv::waitKey(6000);
    cv::destroyAllWindows();

    cv::Mat allHSV;

    cv::cvtColor(allImage, allHSV, cv::COLOR_BGR2HSV);
    cv::imwrite("all_hsv.jpg", allHSV);
    cv::imshow("All_hsv", allHSV);
    cv::waitKey(6000);
    cv::destroyAllWindows();


    cv::Mat allLab;
    cv::cvtColor(allImage,allLab,cv::COLOR_BGR2Lab);
    cv::imwrite("all_lab.jpg",allLab);
    cv::imshow("All_lab", allLab);
    cv::waitKey(6000);
    cv::destroyAllWindows();



    return 0;
}