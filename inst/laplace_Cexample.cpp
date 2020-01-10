#include <RcppEigen.h>
#include <iostream>
#include <fminLaplace.h>
// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using std::pow;
using std::exp;
using std::log; 

class myFun {
public:
  myFun(Eigen::VectorXd y, Eigen::MatrixXd X, Eigen::MatrixXd S) : X_(X), S_(S), y_(y) {}
  double operator()(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const;
private: 
  Eigen::MatrixXd X_; 
  Eigen::MatrixXd S_; 
  Eigen::VectorXd y_; 
};

// [[Rcpp::export]]
double lpdf_norm(const Eigen::VectorXd x, const Eigen::VectorXd mu, const double& sd) {
  double s = (x - mu).dot(x - mu);
  int n = x.size(); 
  double lpdf = -0.5 * n * log(2 * M_PI) - n * log(sd) - 0.5 * s / (sd * sd); 
  return lpdf; 
}

double myFun::operator()(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
  Eigen::VectorXd nupar(x.size() + u.size() - 2); 
  nupar.head(X_.cols() - u.size())= x.tail(x.size() - 2); 
  nupar.tail(u.size()) = u; 
  Eigen::VectorXd nu = X_ * nupar; 
  double llk = lpdf_norm(y_, nu, exp(x(0)));  
  llk += x(1) * S_.cols() * 0.5 - 0.5 * exp(x(1)) * u.transpose() * S_ * u; 
  llk = -llk; 
  return llk;
}

Eigen::VectorXd Perturb(Eigen::VectorXd x, int m, double e, int k = -1, double e2 = 0) {
  Eigen::VectorXd y(x);
  y(m) += e;
  if (k > -1 & k != m) y(k) += e2;
  return(y);
}

// [[Rcpp::export]]
List doF(Eigen::VectorXd x,
         Eigen::VectorXd u, 
         Eigen::VectorXd y, 
         Eigen::MatrixXd X,
         Eigen::MatrixXd S, 
         int maxit = 1000,
         double tol = 1e-7,
         int hessupdate = 10,
         int maxhalfsteps = 10,
         bool verbose = false,
         int digits = 4) {
  myFun F(y, X, S);
  FminLaplace<myFun> fmin(F, x, u, maxit, tol, hessupdate, maxhalfsteps, verbose, digits);
  FminU<myFun> fu(F, x);
  FminMarg<FminU<myFun>> fmarg(fu, u);
  Fmin<FminMarg<FminU<myFun>>> fmins(fmarg, x, 10, 1e-7, 10, 10, true, 4);
  
  int i = 1; 
  double h = 1e-4;
  double f = fmarg(x); 
  Eigen::MatrixXd H1 = fmarg.hessian(); 
 double d0 = fmarg(Perturb(x, i, -2*h));
 Eigen::MatrixXd H2 = fmarg.hessian(); 
  //double d1 = fmarg(Perturb(x, i, -h));
 //double d2 = fmarg(Perturb(x, i, h));
 //double d3 = fmarg(Perturb(x, i, 2*h));
//double g1 = (d3 - d0) / (4 * h);
//double g2 = (d2 - d1) / (2 * h);
 //double grad = (4 * g2 - g1) / 3;
  
  //std::cout << "f = " << f << "d0 = " << d0 << "d1 = " << d1 << "d2 = " << d2 << "d3 = " << d3 << "g1 = " << g1 << "g2 = " << g2 << "g = " << grad << std::endl; 
  
  //std::cout << "g = " << fmins.ComputeG(x) << std::endl;
  
  //fmins.Run(); 
  //std::cout << "fmarg = " << fmarg(x) << std::endl; 
  //std::cout << "u = " << fmarg.u() << std::endl; 
  fmin.Run();
  List res = List::create(Named("x") = fmin.x(),
                          Named("u") = fmin.u(), 
                          Named("H1") = H1,
                          Named("H2") = H2);
  return res;
}



