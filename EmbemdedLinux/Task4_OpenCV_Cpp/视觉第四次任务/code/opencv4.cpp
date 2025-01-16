#include <opencv2/opencv.hpp>
#include <iostream>

// 处理单张图片的函数
void processImage(const std::string& inputPath, const std::string& outputPath) {
    // 读取图片
    cv::Mat frame = cv::imread(inputPath);
    if (frame.empty()) {
        std::cerr << "无法读取图片: " << inputPath << std::endl;
        return;
    }

    // 转换为 HSV 颜色空间
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    // 定义红色的范围
    cv::Scalar lower_red1(0, 120, 70);    // 红色的下限 (低色调)
    cv::Scalar upper_red1(10, 255, 255);  // 红色的上限 (高色调)
    cv::Scalar lower_red2(170, 120, 70);  // 红色的下限 (高色调)
    cv::Scalar upper_red2(180, 255, 255); // 红色的上限 (低色调)

    // 创建两个掩码，用于提取红色
    cv::Mat mask1, mask2, mask;
    cv::inRange(hsv, lower_red1, upper_red1, mask1);
    cv::inRange(hsv, lower_red2, upper_red2, mask2);
    cv::bitwise_or(mask1, mask2, mask);  // 合并两个掩码

    // 使用掩码提取红色区域
    cv::Mat red_area;
    cv::bitwise_and(frame, frame, red_area, mask);

    // 转换为灰度图像进行轮廓检测
    cv::Mat gray;
    cv::cvtColor(red_area, gray, cv::COLOR_BGR2GRAY);

    // 使用高斯模糊去噪
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(blurred, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 创建一个副本，用于绘制结果
    cv::Mat result = frame.clone();

    for (const auto& contour : contours) {
        // 如果轮廓的面积太小，则跳过
        if (cv::contourArea(contour) < 500) {  // 可以根据实际情况调整最小面积
            continue;
        }

        // 计算外接矩形
        cv::Rect rect = cv::boundingRect(contour);

        // 计算外接矩形的几何中心
        cv::Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);

        // 获取轮廓的颜色（与背景相反的颜色）
        cv::Scalar contourColor(255 - frame.at<cv::Vec3b>(center)[0],
                                255 - frame.at<cv::Vec3b>(center)[1],
                                255 - frame.at<cv::Vec3b>(center)[2]);

        // 绘制轮廓
        cv::drawContours(result, std::vector<std::vector<cv::Point>>{contour}, -1, contourColor, 2);

        // 绘制十字标记外接矩形的几何中心（使用与轮廓相同的颜色）
        cv::drawMarker(result, center, contourColor, cv::MARKER_CROSS, 20, 2);
    }

    // 保存处理后的图片
    cv::imwrite(outputPath, result);
    std::cout << "处理后的图片已保存为: " << outputPath << std::endl;
}

int main() {
    // 输入图片路径
    std::string inputImage1 = "wuliao1.jpg";  // 第一张图片路径
    std::string inputImage2 = "wuliao2.jpg";  // 第二张图片路径

    // 输出图片路径
    std::string outputImage1 = "first.jpg";  // 第一张输出图片
    std::string outputImage2 = "second.jpg"; // 第二张输出图片

    // 处理第一张图片
    processImage(inputImage1, outputImage1);

    // 处理第二张图片
    processImage(inputImage2, outputImage2);

    return 0;
}
