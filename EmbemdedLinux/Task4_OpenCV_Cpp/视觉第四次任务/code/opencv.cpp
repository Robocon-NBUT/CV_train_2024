#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Rect roi; // 用于存储选择的矩形区域
Point pt1, pt2; // 鼠标拖动时的两个点
bool drawing = false; // 标志变量，表示是否正在绘制矩形

// 鼠标回调函数
void mouseCallback(int event, int x, int y, int flags, void* userdata) {
    Mat& imgCopy = *(Mat*)userdata; // 引用传递原图副本
    if (event == EVENT_LBUTTONDOWN) {
        // 鼠标左键按下，记录起点
        pt1 = Point(x, y);
        drawing = true;
    }
    else if (event == EVENT_MOUSEMOVE) {
        if (drawing) {
            // 鼠标移动时，实时绘制矩形
            pt2 = Point(x, y);
            imgCopy = imread("/task4/all.jpg"); // 重新加载原图
            rectangle(imgCopy, pt1, pt2, Scalar(0, 255, 0), 2); // 绘制矩形
            imshow("Select Region", imgCopy); // 显示矩形
        }
    }
    else if (event == EVENT_LBUTTONUP) {
        // 鼠标左键抬起，确定矩形区域
        pt2 = Point(x, y);
        drawing = false;
        roi = Rect(pt1, pt2); // 记录矩形区域
        imgCopy = imread("/task4/all.jpg"); // 重新加载原图
        rectangle(imgCopy, pt1, pt2, Scalar(0, 255, 0), 2); // 绘制矩形
        imshow("Select Region", imgCopy); // 显示矩形
    }
}

int main() {
    // 图片路径
    string imagePath = "/task4/all.jpg";

    // 读取图片
    Mat img = imread(imagePath);
    if (img.empty()) {
        cerr << "Error: Cannot load image!" << endl;
        return -1;
    }

    // 创建窗口并设置鼠标回调函数
    namedWindow("Select Region");
    Mat imgCopy = img.clone(); // 创建原图副本
    setMouseCallback("Select Region", mouseCallback, &imgCopy);

    // 显示图片并等待用户选择区域
    imshow("Select Region", imgCopy);
    waitKey(0); // 等待按键

    if (roi.width > 0 && roi.height > 0) {
        // 如果用户选择了一个有效区域，保存该区域
        Mat imgROI = img(roi); // 从原图中截取选定区域
        string outputPath = "/task4/phone.jpg";
        imwrite(outputPath, imgROI); // 保存截取的部分
        cout << "Image saved as " << outputPath << endl;
    } else {
        cout << "No valid region selected." << endl;
    }

    return 0;
}

