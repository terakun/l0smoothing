#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include "./l0smooth.cc"

int main(int argc, char *argv[]){
  if(argc < 2){
    std::cerr << argv[0] << " [source image]" << std::endl;
    return 0;
  }
  cv::Mat src_img = cv::imread(argv[1]);
  if(src_img.empty()) return -1;

  cv::namedWindow("src image",CV_WINDOW_AUTOSIZE);
  cv::imshow("src image", src_img);
  cv::waitKey(0);
  return 0;
}

