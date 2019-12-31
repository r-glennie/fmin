#ifndef fminLaplace_glennie_h
#define fminLaplace_glennie_h

#include <iostream>
#include <Eigen/Dense>
#include "fmin.h"

//// class to compute f(x, u) for a fixed x
template<class Type>
class FminU {
public:
  FminU(Type& fun,
           Eigen::VectorXd x) {
    f_ = fun;
    x_ = x;
  }
  double operator()(const Eigen::VectorXd u) const;
  void setx(const Eigen::VectorXd newx);

private:
  Type f_;
  Eigen::VectorXd x_;
};

double FminU::operator()(const Eigen::VectorXd& u) const {
  return(f_(x_, u));
}

double FminU::setx(const Eigen::VectorXd newx) {x_ = newx;}

//// class to compute marginal f(x)
template<class Type>
class FminMarg {
public:
  FminMarg(Type& fu, const Eigen::VectorXd u) {
    fu_ = fu;
    u_ = u;
  }
  double operator()(const Eigen::VectorXd x) const;
  Eigen::VectorXd u() const;
  Eigen::MatrixXd hessian() const;

private:
  Type fu_;
  Eigen::VectorXd u_;
  Eigen::MatrixXd H_;
};

FminMarg::operator()(const Eigen::VectorX x) const {
  fu_.setx(x);
  Fmin<Type> fmin(fu_, u_, 1000, 1e-7, 0, 10, false, 4);
  u_ = fmin.Par();
  H_ = fmin.ComputeH(u_);
  double val = fu_(u_);
  Eigen::MatrixXd L = H.llt().matrixL();
  double Hldet;
  for (int i = 0; i < L.cols(); ++i) Hldet += log(L(i, i));
  val += Hldet / 2;
  return(val);
}

FminMarg::u() const (return u_;)
FminMarg::hessian() const {return H_;}

//// class to compute optimal parameters from Laplace-approximate marginal likelihood
template <class Type>
class FminLaplace {
public:
  FminLaplace(Type& fun,
       Eigen::VectorXd x,
       Eigen::VectorXd u,
       int maxit = 1000,
       double tol = 1e-7,
       int hessupdate = 30,
       int maxhalfsteps = 10,
       bool verbose = false,
       int digits = 4) {

    x_ = x;
    u_ = u;
    startx_ = x;
    startu_ = u;
    f = fun;
    maxit_ = maxit;
    tol_ = tol;
    hessupdate_ = hessupdate > -1 ? hessupdate : maxit + 1;
    maxhalfsteps_ = maxhalfsteps;
    verbose_ = verbose;
    digits_ = digits;
    conv = 0;
  }

void Run() {
  FminU<Type> fu(f, x_);
  FminMarg<FminU<Type>> fmarg(fu, u_);
  Fmin<FminMarg<FminU<Type>>> fmin(fmarg, x_, maxit_, tol_, hessupdate_, maxhalfsteps_, verbose_, digits_);
  fmin.Run();
  x_ = fmin.Par();
  u_ = fmarg.u();
  fmin_ = fmin;
}

Eigen::VectorXd x() const {return x_;}
Eigen::VectorXd u() const {return u_;}
Fmin<FminMarg<FminU<Type>>> opt const {return fmin_;}


private:
  Eigen::VectorXd x_;
  Eigen::VectorXd u_;
  Eigen::VectorXd startx_;
  Eigen::VectorXd startu_;
  Type f;
  int maxit_;
  double tol_;
  int hessupdate_;
  int maxhalfsteps_;
  bool verbose_;
  int digits_;
  int conv;
  Fmin<FminMarg<FminU<Type>>> fmin_;

};

#endif // fminlaplace_glennie_h



