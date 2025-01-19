#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

// 获取相反颜色的函数
cv::Scalar getOppositeColor(const cv::Scalar& color) {
    return cv::Scalar(255 - color[0], 255 - color[1], 255 - color[2]);
}

// 绘制十字中心的函数
void drawCross(cv::Mat& image, const cv::Point& center, const cv::Scalar& color) {
    int length = 20; // 十字长度
    int thickness = 2; // 十字线的厚度

    cv::line(image, cv::Point(center.x - length / 2, center.y),
             cv::Point(center.x + length / 2, center.y), color, thickness);
    cv::line(image, cv::Point(center.x, center.y - length / 2),
             cv::Point(center.x, center.y + length / 2), color, thickness);
}

int main() {
    // 输入图像路径
    std::string imgPath = "3.jpg";  // 替换为您的图片路径
    std::string outputPath = "output.jpg";  // 保存结果的路径

    // 加载图片
    cv::Mat img = cv::imread(imgPath);
    if (img.empty()) {
        std::cerr << "图像加载失败，请检查路径: " << imgPath << std::endl;
        return -1;
    }

    // 转换为 HSV 颜色空间
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // 定义橙色的 HSV 范围
    cv::Scalar lowerOrange(5, 100, 100); // 橙色最低 HSV 值
    cv::Scalar upperOrange(15, 255, 255); // 橙色最高 HSV 值

    // 定义皮肤的 HSV 范围
    cv::Scalar lowerSkin(0, 30, 60); // 肤色最低 HSV 值
    cv::Scalar upperSkin(20, 150, 255); // 肤色最高 HSV 值

    // 创建橙色掩模
    cv::Mat maskOrange;
    cv::inRange(hsv, lowerOrange, upperOrange, maskOrange);

    // 创建皮肤掩模
    cv::Mat maskSkin;
    cv::inRange(hsv, lowerSkin, upperSkin, maskSkin);

    // 从橙色掩模中排除皮肤区域
    cv::Mat mask;
    cv::bitwise_and(maskOrange, ~maskSkin, mask);

    // 形态学操作去除噪声
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 2);
    cv::morphologyEx(mask, mask, cv::MORPH_DILATE, kernel, cv::Point(-1, -1), 1);

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 创建结果图像
    cv::Mat result = img.clone();

    for (const auto& contour : contours) {
        // 仅处理足够大的轮廓
        double area = cv::contourArea(contour);
        if (area < 500) continue;

        // 绘制轮廓的外接矩形
        cv::Rect boundingBox = cv::boundingRect(contour);
        cv::Point center(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);

        // 获取轮廓的颜色（取平均颜色）
        cv::Mat maskContour = cv::Mat::zeros(img.size(), CV_8UC1);
        cv::drawContours(maskContour, std::vector<std::vector<cv::Point>>{contour}, -1, 255, -1);
        cv::Scalar meanColor = cv::mean(img, maskContour);
        cv::Scalar oppositeColor = getOppositeColor(meanColor);

        // 绘制轮廓
        cv::drawContours(result, std::vector<std::vector<cv::Point>>{contour}, -1, oppositeColor, 2);

        // 绘制几何中心的十字
        drawCross(result, center, oppositeColor);

        // 绘制外接矩形
        cv::rectangle(result, boundingBox, oppositeColor, 2);
    }

    // 保存处理后的图像
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "保存处理后的图像失败!" << std::endl;
        return -1;
    }
    std::cout << "处理后的图像已保存为: " << outputPath << std::endl;

    // 显示结果
    cv::imshow("Original Image", img);
    cv::imshow("Processed Image", result);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
