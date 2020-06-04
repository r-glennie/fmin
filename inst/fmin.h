#ifndef fmin_glennie_h
#define fmin_glennie_h

#include <iostream>
#include <Eigen/Dense>

template <class Type>
class Fmin {
public:
  Fmin(Type& fun,
       Eigen::VectorXd start,
       int maxit = 200,
       double tol = 1e-10,
       double stepmax = 1, 
       int maxsubsteps = 10, 
       bool verbose = false,
       int digits = 4) : f(fun) {

    par = start;
    start_ = start;
    n = par.size();
    maxit_ = maxit;
    stepmax_ = stepmax; 
    tol_ = tol;
    maxsubsteps_ = maxsubsteps;
    verbose_ = verbose;
    digits_ = digits;
    conv = 0;
    alpha = Eigen::VectorXd::Zero(2);
    fvals = Eigen::VectorXd::Zero(2);
  }

  void Run() {
   par = start_;
   // setup loop 
   iter = 0;
   bool loop = true;
   // initial values of f, inverse hessian, and gradient 
   fval = f(par);
   H = Eigen::MatrixXd::Identity(n, n);
   g = ComputeG(par);
   while(loop) {
     ++iter;
     // compute step direction 
     GetNewtonStep(); 
     der = g.dot(newton_step);
     // get stepsize 
     step_size = LineSearch(); 
     // comute step 
     delta = step_size * newton_step; 
     // update parameters
     par += delta; 
     fval = f(par);
     gnew = ComputeG(par);
     // update inverse Hessian 
     BfgsUpdate();
     // update gradient
     g = gnew; 
     // print if wanted 
     if (verbose_) {
       std::cout << std::fixed << std::setprecision(digits_) << fval << "\t|\t" << par.transpose() << std::endl;
     }
     // check stopping criteria 
     if (CheckStop()) loop = false;
   }
   // check convergence 
   if (CheckConv()) conv = 1;
  }

  bool Conv() {return conv;}
  Eigen::VectorXd Par() {return par;}
  int Iter() {return iter;}

  Eigen::VectorXd ComputeG(Eigen::VectorXd x) {
    Eigen::VectorXd y(x);
    Eigen::VectorXd grad(x.size());
    double d0, d1, d2, d3, g1, g2;
    double h = 1e-4;
    for (int i = 0; i < x.size(); ++i) {
      d0 = f(Perturb(x, i, -2*h));
      d1 = f(Perturb(x, i, -h));
      d2 = f(Perturb(x, i, h));
      d3 = f(Perturb(x, i, 2*h));
      g1 = (d3 - d0) / (4 * h);
      g2 = (d2 - d1) / (2 * h);
      grad(i) = (4 * g2 - g1) / 3;
    }
    return(grad);
  }


  Eigen::MatrixXd ComputeH(Eigen::VectorXd x) {
    int n = x.size(); 
    Eigen::MatrixXd H(n, n);
    double f0 = f(x);
    // set initial difference sizes  
    Eigen::MatrixXd h0 = 0.0001 * x;
    for (int i = 0; i < n; ++i) {
      if (h0(i) < 0) h0(i) = -h0(i); 
      if (fabs(x(i)) < 2e-5) h0(i) += 1e-4; 
    }
    // diagonal of H
    Eigen::VectorXd h(n);
    double f1, f2;
    int r = 4; 
    Eigen::VectorXd Hrich(r); 
    for (int i = 0; i < n; ++i) {
      h = h0; 
      for (int k = 0; k < r; ++k) {
        f1 = f(Perturb(x, i, h(i))); 
        f2 = f(Perturb(x, i, -h(i)));
        
        Hrich(k) = (f1 - 2*f0 + f2) / (h(i) * h(i)); 
        h *= 0.5; 
      }
      for (int m = 0; m < r - 1; ++m) {
        for (int k = 0; k < r - m - 1; ++k) {
          Hrich(k) = (Hrich(k + 1) * pow(4, m + 1) - Hrich(k)) / (pow(4, m + 1) - 1); 
        }
      }
      H(i, i) = Hrich(0); 
    }
    // off-diagonal of H
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < i; ++j) {
        h = h0; 
        for (int k = 0; k < r; ++k) {
          f1 = f(Perturb(x, i, h(i), j, h(j))); 
          f2 = f(Perturb(x, i, -h(i), j, -h(j)));
          Hrich(k) = (f1 - 2*f0 + f2 - H(i, i) * h(i) * h(i) - H(j, j) * h(j) * h(j)) / (2 * h(i) * h(j)); 
          h *= 0.5; 
        }
        for (int m = 0; m < r - 1; ++m) {
          for (int k = 0; k < r - m - 1; ++k) {
            Hrich(k) = (Hrich(k + 1) * pow(4, m + 1) - Hrich(k)) / (pow(4, m + 1) - 1); 
          }
        }
        H(i, j) = Hrich(0); 
        H(j, i) = Hrich(0); 
      }
    }
    return(H);
  }

private:
  Type f;
  Eigen::VectorXd start_;
  Eigen::VectorXd par;
  Eigen::VectorXd alpha;
  Eigen::VectorXd fvals;
  Eigen::MatrixXd H;
  double fval;
  double gval; 
  int n;
  double step_size; 
  Eigen::VectorXd delta;
  Eigen::VectorXd newton_step;
  int substeps;
  double der;
  double dernew; 
  Eigen::VectorXd g;
  Eigen::VectorXd gnew;
  int iter;
  double stepmax_; 
  int maxit_;
  double tol_;
  int maxsubsteps_;
  bool verbose_;
  int digits_;
  int conv;

  bool CheckConv() {
    bool stop = false;
    double maxg = fabs(g(0)); 
    double maxstep = fabs(delta(0)); 
    for (int i = 1; i < n; ++i) {
      if (fabs(g(i)) > maxg) maxg = fabs(g(i)); 
      if (fabs(delta(i)) > maxstep) maxstep = fabs(delta(i)); 
    }
    if (maxg < tol_) stop = true;
    if (maxstep < tol_) stop = true;
   return(stop);
  }

  bool CheckStop() {
    bool check_conv = CheckConv();
    if (iter > maxit_) check_conv = true;
    return(check_conv);
  }

  void GetNewtonStep() {
    newton_step = -H * g; 
  }

  double Backtrack(double minstep = 0.1) {
    // quadratic approximation 
    double m, M; 
    if (alpha(0) < alpha(1)) {
      m = alpha(0); 
      M = alpha(1); 
    } else {
      m = alpha(1); 
      M = alpha(0); 
    }
    double diff = alpha(1) - alpha(0); 
    double a = -gval * diff * diff;
    double b = fvals(1) - fvals(0) - diff * gval; 
    double incr = a / (2.0 * b); 
    double newalp = alpha(0) + incr; 
    double r = M - m; 
    if ((newalp - m) / r < minstep) newalp = m + minstep * r; 
    if ((M - newalp) / r < minstep) newalp = M - minstep * r; 
    return(newalp); 
  }

  bool SufficientDecrease(double fnew, double alpha, double c1 = 1e-4) {
    return(fnew <= fval + c1 * alpha * der);
  }

  bool CurvatureOk(double dernew, double c2 = 0.9) {
    return(abs(dernew) <= -c2 * der); 
  }

  double LineSearch() { 
    // start with Newton step 
    alpha(0) = 0; 
    alpha(1) = 1; 
    fvals(0) = fval; 
    fvals(1) = f(par + alpha(1) * newton_step); 
    gval = der; 
    substeps = 1; 
    bool loop = true; 
    bool suff; 
    while (loop) {
      suff = SufficientDecrease(fvals(1), alpha(1)); 
      // if no sufficient decrease anymore, then zoom 
      if (!suff | (substeps > 1 & fvals(1) > fvals(0))) {
        return(Zoom()); 
      }
      gnew = ComputeG(par + alpha(1) * newton_step); 
      dernew = gnew.dot(newton_step); 
      // if curvature is good and sufficient decrease, stop 
      if (CurvatureOk(dernew)) {
        return(alpha(1)); 
      } 
      // if sufficient decrease and positive curvature, zoom 
      if (dernew >= 0) {
        return(Zoom()); 
      }
      // sufficient decrease but poor curvature, extend search 
      alpha(0) = alpha(1); 
      fvals(0) = fvals(1); 
      gval = dernew; 
      alpha(1) = 2 * alpha(1); 
      fvals(1) = f(par + alpha(1) * newton_step); 
      // stop if maximum number of substeps taken 
      if (substeps > maxsubsteps_ | alpha(1) > stepmax_) {
        return(alpha(1)); 
      }
      ++substeps; 
    }
  }

  double Zoom() {
    substeps = 1; 
    double loop = true; 
    double alp; 
    double newf; 
    double oldf = fval;   
    while (loop) {
      alp = Backtrack(); 
      newf = f(par + alp * newton_step); 
      // if no  sufficient decrese make his new step the upper bound
      if (!SufficientDecrease(newf, alp) | newf > oldf) {
        alpha(1) = alp; 
        fvals(1) = newf; 
        oldf = newf; 
      } else {
        gnew = ComputeG(par + alp * newton_step); 
        dernew = gnew.dot(newton_step); 
        if (CurvatureOk(dernew)) {
          return(alp); 
        }
        if (dernew * (alpha(1) - alpha(0)) >= 0) {
          alpha(1) = alpha(0); 
          fvals(1) = fvals(0); 
        }
        alpha(0) = alp;
        fvals(0) = newf; 
        gval = dernew; 
      }
      ++substeps; 
      if (substeps > maxsubsteps_) {
        return(alp);  
      }
    }
    return(alp); 
  }

  void BfgsUpdate() {
    Eigen::VectorXd gdif = gnew - g;
    double r = 1 / gdif.dot(delta); 
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n); 
    H = (I - r * delta * gdif.transpose()) * 
          H * (I - r * gdif * delta.transpose()) + 
            r * delta * delta.transpose(); 
  }

  Eigen::VectorXd Perturb(Eigen::VectorXd x, int m, double e, int k = -1, double e2 = 0) {
    Eigen::VectorXd y(x);
    y(m) += e;
    if (k > -1 & k != m) y(k) += e2;
    return(y);
  }

};
#endif // fmin_glennie_h
