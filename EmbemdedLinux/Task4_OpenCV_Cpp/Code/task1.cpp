#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
int main ()
{
Mat image=imread("/home/task4/all.jpg");
int start_x=164;
int start_y=138;
int width=411;
int height=178;
Mat change=image(Rect(start_x,start_y,width,height));
imshow("image",change);
waitKey(0);
imwrite("/home/task4/phone.jpg",change);
return 0;
}
