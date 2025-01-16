#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

// 获取相反颜色的函数
cv::Scalar getOppositeColor(const cv::Scalar& color) {
    return cv::Scalar(255 - color[0], 255 - color[1], 255 - color[2]);
}

// 在指定中心绘制十字的函数
void drawCross(cv::Mat& img, cv::Point center, const cv::Scalar& color, int size = 10, int thickness = 2) {
    cv::line(img, cv::Point(center.x - size, center.y), cv::Point(center.x + size, center.y), color, thickness);
    cv::line(img, cv::Point(center.x, center.y - size), cv::Point(center.x, center.y + size), color, thickness);
}

int main() {
    // 定义HSV色彩空间中橙色的范围
    cv::Scalar lower_orange(1, 120, 50);   // 下界
    cv::Scalar upper_orange(10, 255, 255); // 上界

    // 处理名称为 "1.jpg" 到 "5.jpg" 的图片
    for(int i = 1; i <= 5; ++i){
        std::string filename = std::to_string(i) + ".jpg";
        cv::Mat frame = cv::imread(filename);

        if(frame.empty()){
            std::cerr << "无法读取文件: " << filename << std::endl;
            continue;
        }

        // 将图像转换为HSV色彩空间
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        // 创建橙色的掩模
        cv::Mat mask;
        cv::inRange(hsv, lower_orange, upper_orange, mask);

        // 使用形态学操作去除噪声
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel, cv::Point(-1,-1), 2);
        cv::morphologyEx(mask, mask, cv::MORPH_DILATE, kernel, cv::Point(-1,-1), 1);

        // 查找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if(!contours.empty()){
            // 假设目标物体是最大的轮廓
            size_t largest_idx = 0;
            double max_area = 0;
            for(size_t j = 0; j < contours.size(); ++j){
                double area = cv::contourArea(contours[j]);
                if(area > max_area){
                    max_area = area;
                    largest_idx = j;
                }
            }

            // 仅在轮廓面积足够大时处理
            if(max_area > 500){
                std::vector<cv::Point> largest_contour = contours[largest_idx];
                cv::Rect bounding_rect = cv::boundingRect(largest_contour);
                cv::Point center(bounding_rect.x + bounding_rect.width / 2, bounding_rect.y + bounding_rect.height / 2);

                // 创建用于计算平均颜色的掩模
                cv::Mat mask_contour = cv::Mat::zeros(frame.size(), CV_8UC1);
                cv::drawContours(mask_contour, contours, largest_idx, cv::Scalar(255), cv::FILLED);
                cv::Scalar mean_val = cv::mean(frame, mask_contour);
                cv::Scalar mean_color(mean_val[0], mean_val[1], mean_val[2]);
                cv::Scalar opposite_color = getOppositeColor(mean_color);

                // 绘制最大的轮廓
                cv::drawContours(frame, contours, largest_idx, opposite_color, 2);

                // 在几何中心绘制十字
                drawCross(frame, center, mean_color);

                // 绘制外接矩形
                cv::rectangle(frame, bounding_rect, opposite_color, 2);
            }
        }

        // 保存处理后的图像
        std::string output_filename = std::to_string(i) + "_change.jpg";
        if(!cv::imwrite(output_filename, frame)){
            std::cerr << "无法保存文件: " << output_filename << std::endl;
        } else {
            std::cout << "已保存: " << output_filename << std::endl;
        }
    }

    return 0;
}
