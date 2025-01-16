#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    // 读取图片
    cv::Mat frame = cv::imread("/root/Desktop/ImageShow/my_venv/photograph/photo.jpg");
    if (frame.empty()) {
        std::cerr << "无法读取图片，请检查文件是否存在及路径是否正确" << std::endl;
        return -1;
    }

    // 设置面积阈值，单位为像素面积，可根据实际情况调整该值
    int area_threshold = 10000;

    // 将图像从 BGR 颜色空间转换为 HSV 颜色空间，方便根据颜色范围筛选橙色
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    // 调整橙色在 HSV 颜色空间中的范围
    cv::Scalar lower_orange(0, 80, 80);
    cv::Scalar upper_orange(30, 255, 255);

    // 根据橙色的 HSV 范围创建掩膜，提取出橙色部分的图像
    cv::Mat mask;
    cv::inRange(hsv, lower_orange, upper_orange, mask);

    // 对提取出的橙色部分图像进行直方图均衡化，增强对比度
    cv::equalizeHist(mask, mask);

    // 使用阈值处理，将图像二值化，这里采用自适应阈值，可根据实际情况调整参数
    cv::Mat thresh;
    cv::adaptiveThreshold(mask, thresh, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 11, 2);

    // 创建结构元素，这里使用矩形结构元素，大小可根据实际调整，例如 (3, 3)
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    // 先进行腐蚀操作，去除小噪点等，迭代次数设为 1（可调整）
    cv::erode(thresh, thresh, kernel, cv::Point(-1, -1), 1);
    // 再进行膨胀操作，恢复一些被腐蚀掉的轮廓部分（如果有的话），迭代次数设为 1（可调整）
    cv::dilate(thresh, thresh, kernel, cv::Point(-1, -1), 1);

    // 查找图像中的轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); ++i) {
        // 计算轮廓的外接矩形
        cv::Rect rect = cv::boundingRect(contours[i]);

        // 计算轮廓的面积
        double contour_area = cv::contourArea(contours[i]);

        // 判断轮廓面积是否大于设定的阈值，如果大于则进行后续操作
        if (contour_area > area_threshold) {
            // 获取外接矩形的几何中心坐标
            int center_x = rect.x + rect.width / 2;
            int center_y = rect.y + rect.height / 2;

            // 获取轮廓区域内某一点的颜色（这里取外接矩形左上角的点作为代表，可根据实际优化）
            cv::Vec3b color = frame.at<cv::Vec3b>(rect.y, rect.x);
            cv::Vec3b reversed_color(255 - color[0], 255 - color[1], 255 - color[2]);

            // 绘制物料轮廓，用反转后的颜色
            cv::drawContours(frame, contours, i, reversed_color, 2);

            // 用与轮廓相同的颜色绘制十字，标记外接矩形的几何中心
            cv::line(frame, cv::Point(center_x - 10, center_y), cv::Point(center_x + 10, center_y), cv::Scalar(color[0], color[1], color[2]), 2);
            cv::line(frame, cv::Point(center_x, center_y - 10), cv::Point(center_x, center_y + 10), cv::Scalar(color[0], color[1], color[2]), 2);

            // 添加绘制矩形框的代码，使用绿色（BGR 格式为 (0, 255, 0)）框住物料，框线宽度为 2
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
        }
    }

    // 保存处理后的图像
    cv::imwrite("processed_photo.jpg", frame);
    std::cout << "处理后的图像已保存为 processed_photo.jpg" << std::endl;

    // 关闭所有打开的窗口
    cv::destroyAllWindows();
    return 0;
}
