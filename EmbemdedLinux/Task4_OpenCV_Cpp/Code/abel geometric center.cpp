#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

void processImage(const std::string& imagePath) {
    
    cv::Mat frame = cv::imread(imagePath);
    if (frame.empty()) {
        std::cout << "Unable to read picture: " << imagePath << std::endl;
        return;
    }

    
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    
    cv::Scalar lowerMaybeRed(0, 100, 150);
    cv::Scalar upperMaybeRed(20, 255, 255);

    
    cv::Mat maskRed;
    cv::inRange(hsv, lowerMaybeRed, upperMaybeRed, maskRed);
    
    cv::medianBlur(maskRed, maskRed, 5);
    
    cv::GaussianBlur(maskRed, maskRed, cv::Size(3, 3), 0);

    
    cv::Mat resRed;
    cv::bitwise_and(frame, frame, resRed, maskRed);

    
    cv::Mat grayRed;
    cv::cvtColor(resRed, grayRed, cv::COLOR_BGR2GRAY);
    cv::Mat edgesRed;
    cv::Canny(grayRed, edgesRed, 50, 150);

    
    std::vector<std::vector<cv::Point>> contoursRed;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edgesRed, contoursRed, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contoursRed) {
        double area = cv::contourArea(contour);
        
        if (area > 50) {
            
            cv::Rect boundingRect = cv::boundingRect(contour);
            cv::Mat region = frame(boundingRect);
            if (!region.empty()) {
                
                cv::Scalar meanColor = cv::mean(region);
                cv::Scalar oppositeColor(255 - meanColor[0], 255 - meanColor[1], 255 - meanColor[2]);

               
                cv::drawContours(frame, std::vector<std::vector<cv::Point>>{contour}, 0, oppositeColor, 2);

                
                cv::RotatedRect rect = cv::minAreaRect(contour);
                std::vector<cv::Point2f> boxPoints;
                rect.points(boxPoints);
                std::vector<cv::Point> box;
                for (const auto& point : boxPoints) {
                    box.emplace_back(cv::Point(cvRound(point.x), cvRound(point.y)));
                }

                
                cv::Point center(
                    cvRound((box[0].x + box[2].x) / 2),
                    cvRound((box[0].y + box[2].y) / 2)
                );

                
                cv::line(frame, cv::Point(center.x - 10, center.y), cv::Point(center.x + 10, center.y), oppositeColor, 2);
                cv::line(frame, cv::Point(center.x, center.y - 10), cv::Point(center.x, center.y + 10), oppositeColor, 2);
            }
        }
    }

    
    std::string outputPath = "processed_" + imagePath.substr(imagePath.find_last_of('/') + 1);
    if (cv::imwrite(outputPath, frame)) {
        std::cout << "picture " << outputPath << " normal storage" << std::endl;
    } else {
        std::cout << "Unable to save picture " << outputPath << std::endl;
    }
}

int main() {
    const std::string imagePaths[] = {"/opencv.cpp/picture1.jpg", "/opencv.cpp/picture2.jpg", "/opencv.cpp/picture3.jpg"};
    for (const auto& path : imagePaths) {
        processImage(path);
    }
    return 0;
} docker cp container_name:/opencv.cpp/phone.jpg D:/机器视觉/opencv.cpp/picture/