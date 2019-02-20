#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <unsupported/Eigen/FFT>
#include <complex>
#include <iostream>
#include <chrono>

#include "./l0smooth.h"


void l0smoothing::operator()(const cv::Mat &src_img,cv::Mat &dst_img){
  if( src_img.channels() == 3 ){
    std::vector<cv::Mat> bgr_imgs;
    std::vector<cv::Mat> smoothed_bgr_imgs;
    cv::split(src_img,bgr_imgs);
    for(int i=0;i<3;++i){
      cv::Mat dst_gray_img;
      compute_gray_img(bgr_imgs[i],dst_gray_img);
      smoothed_bgr_imgs.push_back(dst_gray_img);
    }
    cv::merge(smoothed_bgr_imgs,dst_img);
  }else{
    compute_gray_img(src_img,dst_img);
  }
}

void l0smoothing::compute_gray_img(const cv::Mat &src_img,cv::Mat &dst_img){
  img_rows_ = src_img.rows;
  img_cols_ = src_img.cols;
  std::cout << img_rows_ << "," << img_cols_ << std::endl;

  horizontal_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);
  vertical_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);

  horizontal_fft_ = Eigen::MatrixXcd::Zero(img_rows_,img_cols_);
  vertical_fft_ = Eigen::MatrixXcd::Zero(img_rows_,img_cols_);

  impulse_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);
  D_horizontal_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);
  D_vertical_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);

  impulse_fft_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);
  D_horizontal_fft_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);
  D_vertical_fft_ = Eigen::MatrixXd::Zero(img_rows_,img_cols_);

  denom_fft_ = Eigen::MatrixXcd::Zero(img_rows_,img_cols_);

  src_fft_ = Eigen::MatrixXcd::Zero(img_rows_,img_cols_);

  impulse_(0,0) = 1;

  D_horizontal_(0,0) = -1;
  D_horizontal_(0,img_cols_-1) = 1;

  D_vertical_(0,0) = -1;
  D_vertical_(img_rows_-1,0) = 1;
 
  fft_2dim(impulse_fft_,impulse_);
  fft_2dim(D_horizontal_fft_,D_horizontal_);
  fft_2dim(D_vertical_fft_,D_vertical_);

  cv::cv2eigen(src_img,src_img_);
  src_img_ /= 255.0;

  fft_2dim(src_fft_,src_img_);
  s_ = src_img_;

  beta_ = beta0_;
  while( beta_ <= max_beta_ ){
    denom_fft_ = impulse_fft_ + beta_*(D_horizontal_fft_.conjugate().cwiseProduct(D_horizontal_fft_) + D_vertical_fft_.conjugate().cwiseProduct(D_vertical_fft_));

    // solve for h_p^{(i)} and v_p^{(i)}
    for(int r=0;r<img_rows_;++r){
      for(int c=0;c<img_cols_;++c){
        Eigen::Vector2d grad = compute_grad(r,c,s_);
        if( grad.squaredNorm() <= lambda_/beta_ ){
          horizontal_(r,c) = 0;
          vertical_(r,c) = 0;
        }else{
          horizontal_(r,c) = grad[0];
          vertical_(r,c) = grad[1];
        }
      }
    }
    // solve for S^{(i+1)}

    fft_2dim(horizontal_fft_,horizontal_);
    fft_2dim(vertical_fft_,vertical_);

    Eigen::MatrixXcd s_fft(img_rows_,img_cols_);

    Eigen::MatrixXcd numer_fft = src_fft_ + beta_*(D_horizontal_fft_.conjugate().cwiseProduct(horizontal_fft_) + D_vertical_fft_.conjugate().cwiseProduct(vertical_fft_));
    s_fft = numer_fft.cwiseQuotient(denom_fft_);
    Eigen::MatrixXcd s_complex(img_rows_,img_cols_);

    fft_2dim(s_complex,s_fft,false);
    s_ = s_complex.real();

    beta_ *= kappa_;

    cv::Mat tmp_img;
    cv::eigen2cv(s_,tmp_img);
    tmp_img.convertTo(dst_img,CV_8UC1,255);
    // cv::imshow("processing",dst_img);
    // cv::waitKey(1);
    std::cout << "beta:" << beta_ << std::endl;

  }
  
  // Eigen::MatrixXd grad_img(img_rows_,img_cols_);
  // for(int r=0;r<img_rows_;++r){
  //   for(int c=0;c<img_cols_;++c){
  //     double h = horizontal_(r,c) , v = vertical_(r,c);
  //     grad_img(r,c) = std::sqrt(h*h+v*v);
  //   }
  // }
  // cv::Mat tmp_img,grad_cv_img;
  // cv::eigen2cv(grad_img,tmp_img);
  // tmp_img.convertTo(grad_cv_img,CV_8UC1,255);
  // grad_cv_img = 255 - grad_cv_img;
  // cv::imshow("grad",grad_cv_img);
}


void l0smoothing::fft_2dim(Eigen::MatrixXcd &dst_mat,const Eigen::MatrixXcd &src_mat,bool forward)const{
  Eigen::FFT<double> fft;
  int rows = src_mat.rows();
  int cols = src_mat.cols();

  Eigen::MatrixXcd work_mat(rows,cols);
  work_mat = src_mat;
  
  Eigen::MatrixXcd tmp_mat(rows,cols);
  for(int c=0;c<cols;++c){
    Eigen::VectorXcd src_vec = work_mat.col(c);
    Eigen::VectorXcd tmp_vec(rows);
    if(forward){
      fft.fwd(tmp_vec,src_vec);
    }else{
      fft.inv(tmp_vec,src_vec);
    }
    tmp_mat.col(c) = tmp_vec;
  }

  for(int r=0;r<rows;++r){
    Eigen::VectorXcd tmp_vec = tmp_mat.row(r);
    Eigen::VectorXcd dst_vec(cols);
    if(forward){
      fft.fwd(dst_vec,tmp_vec);
    }else{
      fft.inv(dst_vec,tmp_vec);
    }
    dst_mat.row(r) = dst_vec;
  }
}

