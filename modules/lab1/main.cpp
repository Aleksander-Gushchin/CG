#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iostream>

void show(std::string name, const cv::Mat & img) {
  cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
  cv::imshow(name, img);
  //cv::waitKey(0);
}


int main(){
  cv::Mat img = cv::imread("D:/Git/CG/images/h3.jpg");

  show("Original", img);

  cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
  show("Gray", img);
  cv::Mat gray_img = img.clone();

  cv::equalizeHist(img, img);
  show("Equalize", img);

  cv::Canny(img, img, 100, 275);
  show("Canny", img);
  cv::Mat img_edge = img.clone();
  cv::Mat img_corner = cv::Mat::zeros(img.size(), cv::CMP_EQ);

  cv::cornerHarris(img, img_corner, 5, 5, CV_8UC1);

  cv::Mat img_circle = img.clone();
  float thr = static_cast<float>(200) / 256;
  for (size_t i = 0; i < img.rows; ++i)
    for (size_t j = 0; j < img.cols; ++j)
      if (img_corner.at<float>(i, j) > thr)
        cv::circle(img_circle, cv::Point(j, i), 2, cv::Scalar(255), 1, 8, 0);

  //show("Harris", img_corner);

  show("Harris detection", img_circle);

  img_edge.convertTo(img, CV_8UC1, 255, 0);
  //show("Harris2", img);
  cv::bitwise_not(img, img);
  cv::Mat distance_mat;
  cv::distanceTransform(img, distance_mat, cv::DIST_L2, 3);

  cv::Mat integral_img;
  cv::integral(gray_img, integral_img);
  cv::Mat final_img = cv::Mat::zeros(img.size(), CV_8UC1);
  for (size_t i = 0; i < img.rows; ++i)
    for (size_t j = 0; j < img.cols; ++j) {
      int shift = (int)(0,1 * distance_mat.at<float>(i, j));
      int h1 = std::max(0, (int)i - shift);
      int h2 = std::min((int)i + shift, img.rows - 1);
      int w1 = std::max(0, (int)j - shift);
      int w2 = std::min((int)j + shift, img.cols - 1);
      
      int A = integral_img.at<int>((h1 == 0 ? 0 : h1 - 1), (w1 == 0 ? 0 : w1 - 1)); //left up
      int B = integral_img.at<int>((h1 == 0 ? 0 : h1 - 1), w2); //right up
      int C = integral_img.at<int>(h2, (w1 == 0 ? 0 : w1 - 1)); //left down
      int D = integral_img.at<int>(h2, w2); //right down

      int square = ((h2 - h1 + 1) * (w2 - w1 + 1));
      final_img.at<uchar>(i, j) = (D + A - C - B) / square;
      //if ((D + A - C - B) / square > 255)
      //  std::cout << "i: " << i << " j:" << j << " I: " << (D + A - C - B) / square << "\n";
    }


  show("Mean", final_img);
  cv::waitKey(0);
  return 0;
}