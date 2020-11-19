#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

void show(std::string name, cv::Mat img) {
  cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
  cv::imshow(name, img);
  cv::waitKey(0);
}

int main(){
  //std::string image_path = cv::samples::findFile("starry_night.jpg");
  cv::Mat img = cv::imread("D:/Git/CG/images/h3.jpg");

  show("original", img);

  cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
  show("gray", img);

  cv::equalizeHist(img, img);
  show("equalize", img);

  cv::Canny(img, img, 125, 255);
  show("Canny", img);

  return 0;
}