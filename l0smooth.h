#ifndef L0SMOOTH_H
#define L0SMOOTH_H
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

class l0smoothing{
  double lambda_,beta_,beta0_,max_beta_,kappa_;
  int img_rows_,img_cols_;

  Eigen::MatrixXd src_img_,s_;
  Eigen::MatrixXcd src_fft_;

  Eigen::MatrixXd horizontal_,vertical_;
  Eigen::MatrixXcd horizontal_fft_,vertical_fft_;

  Eigen::MatrixXd impulse_;
  Eigen::MatrixXcd impulse_fft_;
  Eigen::MatrixXd D_horizontal_,D_vertical_;
  Eigen::MatrixXcd D_horizontal_fft_,D_vertical_fft_;

  Eigen::MatrixXcd denom_fft_;

  int max_cnt_;
  double epsilon_;
  public:

  void fft_2dim(Eigen::MatrixXcd &dst_mat,const Eigen::MatrixXcd &src_mat,bool forward=true)const;

  void set_epsilon(double e){ epsilon_ = e; }
  void set_max_count(int mc){ max_cnt_ = mc; }
  void set_beta0(double b0){ beta0_ = b0; }
  void set_max_beta(double maxb){ max_beta_ = maxb; }
  void set_kappa(double kappa){ kappa_ = kappa; }
  void set_lambda(double lambda){ lambda_ = lambda; }

  void operator()(const cv::Mat &src_img,cv::Mat &dst_img);
  void compute_gray_img(const cv::Mat &src_img,cv::Mat &dst_img);

  Eigen::Vector2d compute_grad(int r,int c,const Eigen::MatrixXd &mat)const{
    int rows = mat.rows();
    int cols = mat.cols();
    Eigen::Vector2d grad;
    grad << mat(r,(c+1)%cols) - mat(r,c) , mat((r+1)%rows,c) - mat(r,c);
    return grad;
  }

};

#endif
