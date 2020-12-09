#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <omp.h>

#include <random>
#include <algorithm>
#include <iostream>

std::mt19937 gen(time(0));

double activation(double d) {
  return std::max(0.0, d);
}

void fillKernel(cv::Mat & kernel) {
  double sum = 0;
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        kernel.at<cv::Vec3d>(i, j)[c] = static_cast<double>(gen() % 100) / 100;
        sum += kernel.at<cv::Vec3d>(i, j)[c];
      }
    }
  }
  sum /= (kernel.rows * kernel.cols * kernel.channels());
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        kernel.at<cv::Vec3d>(i, j)[c] -= sum;
      }
    }
  }
}

void pooling(cv::Mat& src, cv::Mat& dst) {
  for(int i = 0; i < src.rows / 2; ++i)
    for (int j = 0; j < src.cols / 2; ++j) {
      int a = std::max(src.at<uint16_t>(2 * i, 2 * j), src.at<uint16_t>(2 * i, 2 * j + 1));
      int b = std::max(src.at<uint16_t>(2 * i + 1, 2 * j), src.at<uint16_t>(2 * i + 1, 2 * j + 1));
      dst.at<uint16_t>(i, j) = std::max(a, b);
    }
}

int main(){
  cv::Mat img = cv::imread("D:/Git/CG/images/g1.jpg");

  cv::Mat kernel[5];
  for (int i = 0; i < 5; ++i) {
    kernel[i] = cv::Mat::zeros(3, 3, CV_64FC3);
    fillKernel(kernel[i]);
  }

  int r = kernel[0].cols / 2;

  cv::Mat result[5];
  for (int i = 0; i < 5; ++i) {
    result[i] = cv::Mat::zeros(img.rows - 2 * r, img.cols - 2 * r, CV_16UC1);
  }

  cv::Mat result_pool[5];
  for (int i = 0; i < 5; ++i) {
    result_pool[i] = cv::Mat::zeros(result[0].rows / 2, result[0].cols / 2, CV_16UC1);
  }
  
  for (int k = 0; k < 5; ++k) {
    #pragma omp parallel for num_threads(12)
    for (int i = r; i < img.rows - r; ++i)
      for (int j = r; j < img.cols - r; ++j) {
        double sum = 0;

        for (int l = -r; l <= r; ++l)
          for (int m = -r; m <= r; ++m)
            for (int c = 0; c < 3; ++c)
              sum += img.at<cv::Vec3b>(i + m, j + l)[c] * kernel[k].at<cv::Vec3d>(l + r, m + r)[c];

        result[k].at<uint16_t>(i - r, j - r) = static_cast<uint16_t>(activation(sum));
      }
  }

  for (int i = 0; i < 5; ++i) {
    pooling(result[i], result_pool[i]);
  }
  for (int k = 0; k < 5; ++k)
    for (int i = 0; i < result_pool[k].rows; ++i)
      for (int j = 0; j < result_pool[k].cols; ++j)
        if (result_pool[k].at<uint16_t>(i, j) > 255)
          std::cout << result_pool[k].at<uint16_t>(i, j) << "\n";

  return 0;
}