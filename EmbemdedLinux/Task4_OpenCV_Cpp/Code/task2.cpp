#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
int main ()
{
Mat all=imread("/home/task4/all.jpg");
Mat phone1=imread("/home/task4/phone.jpg");
Size size1=all.size();
int width1=size1.width;
int height1=size1.height;
Mat phone2;
resize(phone1,phone2,Size(width1,height1));
imshow("phone",phone2);
waitKey(0);
imwrite("/home/task4/phone_resized.jpg",phone2);
}
