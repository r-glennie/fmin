#include <RcppEigen.h>
#include <iostream>
#include <fmin.h>
// [[Rcpp::depends(RcppEigen)]]

using std::pow; 

class myFun {
public:
  myFun() {}
  double operator()(const Eigen::VectorXd& x) const;
};

double myFun::operator()(const Eigen::VectorXd& x) const {
  double val = 100 * pow(x(1) - x(0)*x(0), 2) + pow(1 - x(0), 2); 
  return val;
}

// [[Rcpp::export]]
void doF() {
  myFun F;
  Eigen::VectorXd x(2);
  x(0) = 1; x(1) = 2;
  Fmin<myFun> fmin(F, x);
  fmin.Run();
  std::cout << "x = " << fmin.Par() << std::endl;
}



