#include <RcppEigen.h>
#include <iostream>
#include <fmin.h>
// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using std::pow;

class myFun {
public:
  myFun() {}
  double operator()(const Eigen::VectorXd& x) const;
};

double myFun::operator()(const Eigen::VectorXd& x) const {
  double val = 100 * pow(x(1) - x(0) * x(0), 2) + pow(1 - x(0), 2);
  //for (int i = 0; i < x.size(); ++i) val += x(i) * x(i);
  return val;
}

// [[Rcpp::export]]
List doF(Eigen::VectorXd x,
         int maxit = 1000,
         double tol = 1e-7,
         int hessupdate = 10,
         int maxhalfsteps = 10,
         bool verbose = false,
         int digits = 4) {
  myFun F;
  Fmin<myFun> fmin(F, x, maxit, tol, hessupdate, maxhalfsteps, verbose, digits);
  fmin.Run();
  List res = List::create(Named("par") = fmin.Par(),
                          Named("value") = F(fmin.Par()),
                          Named("g") = fmin.ComputeG(fmin.Par()),
                          Named("H") = fmin.ComputeH(fmin.Par()),
                          Named("conv") = fmin.Conv(),
                          Named("niter") = fmin.Iter());
  return res;
}



