#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat image = cv::imread("all.jpg");

    int x = 18;
    int y = 90;
    int width = 620;
    int height = 310;


    cv::Mat phone = image(cv::Rect(x, y, width, height));

    cv::imwrite("phone.jpg", phone);

    //cv::imshow("All Image", image);
    cv::imshow("Phone Image",phone);

    cv::waitKey(0);

    std::cout << "Image cropped and saved successfully!" << std::endl;
    return 0;
}