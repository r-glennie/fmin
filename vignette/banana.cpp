#include <RcppEigen.h>
#include <iostream>
#include <fmin.h>
// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;

// Define a class with an () operator 
class myFun {
public:
  myFun() {}
  double operator()(const Eigen::VectorXd& x) const;
};

// Define function I want to minimise 
double myFun::operator()(const Eigen::VectorXd& x) const {
  double f = 0; 
  f += (1 - x[0]) * (1 - x[0]); 
  f += 100 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]); 
  return f;
}

// [[Rcpp::export]]
List doF(Eigen::VectorXd x,
         int maxit = 1000,
         double tol = 1e-10,
         double stepmax = 1, 
         int maxsubsteps = 10,
         bool verbose = false,
         int digits = 4) {
  
  // create an instance of my function 
  myFun F;
  // create instance of the fmin engine 
  Fmin<myFun> fmin(F, x, maxit, tol, stepmax, maxsubsteps, verbose, digits);
  // run fmin on my function to find minimum 
  fmin.Run();
  // package up results to return to R
  List res = List::create(Named("par") = fmin.Par(),
                          Named("value") = F(fmin.Par()),
                          Named("g") = fmin.ComputeG(fmin.Par()),
                          Named("H") = fmin.ComputeH(fmin.Par()),
                          Named("conv") = fmin.Conv(),
                          Named("niter") = fmin.Iter());
  return res;
}


