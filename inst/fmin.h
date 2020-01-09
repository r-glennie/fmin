#ifndef fmin_glennie_h
#define fmin_glennie_h

#include <iostream>
#include <Eigen/Dense>

template <class Type>
class Fmin {
public:
  Fmin(Type& fun,
       Eigen::VectorXd start,
       int maxit = 1000,
       double tol = 1e-7,
       int hessupdate = 30,
       int maxhalfsteps = 10,
       bool verbose = false,
       int digits = 4) : f(fun) {

    par = start;
    start_ = start;
    n = par.size();
    maxit_ = maxit;
    tol_ = tol;
    hessupdate_ = hessupdate > -1 ? hessupdate : maxit + 1;
    maxhalfsteps_ = maxhalfsteps;
    verbose_ = verbose;
    digits_ = digits;
    conv = 0;
    alpha = Eigen::VectorXd::Zero(2);
    fvals = Eigen::VectorXd::Zero(2);
  }

  void Run() {
   par = start_;
   iter = 0;
   bool loop = true;
   fval = f(par);
   Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
   Eigen::LDLT<Eigen::MatrixXd> IR(I);
   R = IR;
   g = ComputeG(par);
   double gval;
   Eigen::MatrixXd H(n, n);
   int lastupdate = 0;
   while(loop) {
     ++iter;
     ++lastupdate;
     if (lastupdate > hessupdate_) {
       lastupdate = 0;
       H = ComputeH(par);
       H = MakePositiveDefinite(H);
       if(CheckPositiveDefinite(H)) R = Eigen::LDLT<Eigen::MatrixXd>(H);
     }
     GetNewtonStep();
     halfsteps = 0;
     gval = 1e-4 * newton_step.dot(g);
     der = g.dot(newton_step);
     alpha(0) = 1;
     alpha(1) = 0;
     fvals(0) = fval;
     fvals(1) = f(par + newton_step);
     delta = newton_step;
     while(f(par + delta) > fval + gval & halfsteps < maxhalfsteps_) {
       GetHalfStep();
       gval = 1e-4 * alpha(1) * der;
       delta = alpha(1) * newton_step;
       ++halfsteps;
     }
     par += delta;
     fval = f(par);
     if (verbose_) {
       std::cout << std::fixed << std::setprecision(digits_) << fval << "\t|\t" << par.transpose() << std::endl;

     }
     gnew = ComputeG(par);
     BfgsUpdate();
     g = gnew;
     if (CheckStop()) loop = false;
   }
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
    Eigen::MatrixXd h0 = 0.001 * x;
    for (int i = 0; i < n; ++i) {
      if (x(i) < 2e-5) h0(i) = 1e-4; 
      if (h0(i) < 0) h0(i) = -h0(i); 
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
        for (int k = 0; k < r - m; ++k) {
          Hrich(k) = (Hrich(k + 1) * pow(4, m + 1) - Hrich(k)) / (pow(4, m + 1)- 1); 
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
          for (int k = 0; k < r - m; ++k) {
            Hrich(k) = (Hrich(k + 1) * pow(4, m + 1) - Hrich(k)) / (pow(4, m + 1)- 1); 
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
  Eigen::LDLT<Eigen::MatrixXd> R;
  double fval;
  int n;
  Eigen::VectorXd delta;
  Eigen::VectorXd newton_step;
  int halfsteps;
  double der;
  Eigen::VectorXd g;
  Eigen::VectorXd gnew;
  int iter;
  int maxit_;
  double tol_;
  int hessupdate_;
  int maxhalfsteps_;
  bool verbose_;
  int digits_;
  int conv;

  bool CheckPositiveDefinite(Eigen::MatrixXd X) {
    bool pos = true;
    Eigen::LLT<Eigen::MatrixXd> R(X);
    if(R.info() == Eigen::NumericalIssue) {
        pos = false;
    }
    return(pos);
  }

  Eigen::MatrixXd MakePositiveDefinite(Eigen::MatrixXd X) {
    Eigen::MatrixXd Y = X;
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(X.rows(), X.cols());
    I *= 1e-7;
    int iter = 0;
    while (!CheckPositiveDefinite(Y)) {
      ++iter;
      Y += I;
      if (iter > 100) {
        //std::cerr << "Cannot make Hessian positive-definite. Increase hessupdate." << std::endl;
        break;
      }
    }
    return(Y);
  }

  bool CheckConv() {
    bool stop = false;
    double maxg = fabs(g(0) / (par(0) + 1e-10));
    double maxstep = fabs(delta(0) / (par(0) + 1e-10));
    double relg;
    double relstep;
    for (int i = 1; i < n; ++i) {
      relg =  fabs(g(i) / (par(i) + 1e-10));
      if (relg > maxg) maxg = relg;
      relstep = fabs(delta(i) / (par(i) + 1e-10));
      if (relstep > maxstep) maxstep = relstep;
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
    newton_step = R.solve(-g);
    if (verbose_) std::cout << "g = " << g << " step = " << newton_step << std::endl; 
  }

  void GetHalfStep() {
    if (halfsteps == 0) {
      alpha(1) = -der / (2.0 * (fvals(1) - fvals(0) - der));
      fvals(1) = f(par + alpha(1) * newton_step);
    } else {
      Eigen::VectorXd v(2);
      v(0) = fvals(1) - fval - der * alpha(1);
      v(1) = fvals(0) - fval - der * alpha(0);
      double a = v(0) * alpha(0) * alpha(0) - v(1) * alpha(1) * alpha(1);
      double b = -v(0) * pow(alpha(0), 3) + v(1) * pow(alpha(1), 3);
      double c = alpha(0)  * alpha(0) * alpha(1) * alpha(1) * (alpha(1) - alpha(0));
      a = a / c;
      b = b / c;
      alpha(0) = alpha(1);
      fvals(0) = fvals(1);
      alpha(1) = (-b + sqrt(b*b - 3 * a * der)) / (3.0 * a);
      fvals(1) = f(par + alpha(1) * newton_step);
    }
    if (alpha(1) > 0.5 * alpha(0)) alpha(1) = 0.5 * alpha(0);
    if (alpha(1) < alpha(0) * 0.1) alpha(1) = 0.1 * alpha(0);
  }

  void BfgsUpdate() {
    Eigen::VectorXd gdif = gnew - g;
    Eigen::VectorXd down = gdif / sqrt(abs(1e-10 + delta.transpose() * gdif));
    Eigen::VectorXd up = gdif / sqrt(abs(1e-10 + gdif.transpose() * delta));
    R = R.rankUpdate(up).rankUpdate(down, -1);
  }

  Eigen::VectorXd Perturb(Eigen::VectorXd x, int m, double e, int k = -1, double e2 = 0) {
    Eigen::VectorXd y(x);
    y(m) += e;
    if (k > -1 & k != m) y(k) += e2;
    return(y);
  }





};



#endif // fmin_glennie_h



