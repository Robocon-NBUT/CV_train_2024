#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
int main()
{
Mat all=imread("/home/task4/all.jpg");
Mat all_gray;
Mat all_hsv;
Mat all_lab;
cvtColor(all,all_gray,COLOR_BGR2GRAY);
cvtColor(all,all_hsv,COLOR_BGR2HSV);
cvtColor(all,all_lab,COLOR_BGR2Lab);
imshow("gray",all_gray);
waitKey(0);
imwrite("/home/task4/all_gray.jpg",all_gray);
imshow("hsv",all_hsv);
waitKey(0);
imwrite("/home/task4/all_hsv.jpg",all_hsv);
imshow("lab",all_lab);
waitKey(0);
imwrite("/home/task4/all_lab.jpg",all_lab);
}
