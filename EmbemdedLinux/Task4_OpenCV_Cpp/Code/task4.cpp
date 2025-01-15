#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>


using namespace std;
using namespace cv;


// 处理图像的函数
void handle(Mat img, int i)
{
    // 将图像从 BGR 转换为 Lab 颜色空间
    Mat lab;
    cvtColor(img, lab, COLOR_BGR2Lab);


    // 定义橙色的 LAB 范围
    Scalar lower_orange(14, 140, 154);  // LAB 下限 (L, A, B)
    Scalar upper_orange(255, 180, 200);  // LAB 上限 (L, A, B)


    // 创建掩膜，仅保留橙色区域
    Mat mask;
    inRange(lab, lower_orange, upper_orange, mask);


    // 使用高斯模糊去噪
    GaussianBlur(mask, mask, Size(5, 5), 0);


    // 查找掩膜上的轮廓
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);


    if (!contours.empty()) {
        // 获取最大轮廓
        auto largest_contour = *max_element(contours.begin(), contours.end(), [](const vector<Point>& c1, const vector<Point>& c2) {
            return contourArea(c1) < contourArea(c2);
        });


        if (contourArea(largest_contour) > 1000) {
            // 获取轮廓的外接矩形
            Rect rect = boundingRect(largest_contour);
            int x = rect.x;
            int y = rect.y;
            int w = rect.width;
            int h = rect.height;


            // 计算外接矩形的中心点
            int center_x = x + w / 2;
            int center_y = y + h / 2;


            Vec3b point_color = img.at<Vec3b>(center_y, center_x);
            Vec3b point_color1;
            for (int i = 0; i < 3; ++i) {
                // 将 0 和 255 都转换为 unsigned char 类型
                point_color1[i] = max(static_cast<unsigned char>(0), min(static_cast<unsigned char>(255), point_color[i]));
            }


            Vec3b inverted_color;
            for (int i = 0; i < 3; ++i) {
                inverted_color[i] = 255 - point_color1[i];
            }


            Scalar contour_color(inverted_color[0], inverted_color[1], inverted_color[2]);


            // 绘制轮廓边缘
            drawContours(img, vector<vector<Point>>{largest_contour}, -1, contour_color, 2);
            rectangle(img, Point(x, y), Point(x + w, y + h), contour_color, 2);
            int cross_size = 10;
            line(img, Point(center_x - cross_size, center_y), Point(center_x + cross_size, center_y), contour_color, 2);
            line(img, Point(center_x, center_y - cross_size), Point(center_x, center_y + cross_size), contour_color, 2);
        }
    }


    // 显示处理后的图像
    imshow("Processed Image", img);
    // 使用 std::string 和 std::to_string 构造文件名
    string filename = "/home/task4/finish" + std::to_string(i) + ".jpg";
    imwrite(filename, img);
    waitKey(0);
}


int main()
{
    Mat img1 = imread("/home/task4/frame1.jpg");
    Mat img2 = imread("/home/task4/frame2.jpg");
    Mat img3 = imread("/home/task4/frame3.jpg");

    handle(img1, 1);
    handle(img2, 2);
    handle(img3, 3);


    return 0;
}