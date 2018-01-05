#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include "./l0smooth.h"

int main(int argc, char *argv[]){
  if(argc < 3){
    std::cerr << argv[0] << " [source image] [lambda]" << std::endl;
    return 0;
  }
  cv::Mat src_img = cv::imread(argv[1],0);
  if(src_img.empty()) return -1;

  l0smoothing smoother;
  double lambda = std::stod(argv[2]);
  double beta0 = 2.0*lambda;
  double max_beta = 1.0e5;
  double kappa = 2.0;
  smoother.set_lambda(lambda);
  smoother.set_beta0(beta0);
  smoother.set_max_beta(max_beta);
  smoother.set_kappa(kappa);

  cv::Mat dst_img;
  smoother(src_img,dst_img);

  cv::namedWindow("src image",CV_WINDOW_AUTOSIZE);
  cv::imshow("src image", src_img);

  cv::namedWindow("dst image",CV_WINDOW_AUTOSIZE);
  cv::imshow("dst image", dst_img);

  cv::waitKey(0);
  return 0;
}

