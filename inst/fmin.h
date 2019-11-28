#ifndef fmin_glennie_h
#define fmin_glennie_h

#include <iostream>
#include <Eigen/Dense>

template <class Type>
class Fmin {
public:
  Fmin(Type& fun,
       Eigen::VectorXd start,
       int maxit = 10,
       double tol = 1e-7,
       int hessupdate = 10,
       int maxhalfsteps = 10,
       bool verbose = false,
       int digits = 4) {

    par = start;
    start_ = start; 
    n = par.size(); 
    f = fun;
    maxit_ = maxit;
    tol_ = tol;
    hessupdate_ = hessupdate;
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
   while(loop) {
     ++iter;
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
     std::cout << "g = " << g << std::endl; 
     fval = f(par);
     gnew = ComputeG(par); 
     BfgsUpdate();
     g = gnew;
     if (CheckStop()) loop = false; 
   }
   if (CheckConv()) conv = 1;  
  }

  bool Conv() {return conv;} 
  Eigen::VectorXd Par() {return par;} 
   

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
        std::cerr << "Cannot make Hessian positive-definite." << std::endl;
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

  Eigen::VectorXd ComputeG(Eigen::VectorXd x) {
    Eigen::VectorXd y(n); 
    Eigen::VectorXd grad(n);
    double f0 = f(x); 
    for (int i = 0; i < n; ++i) y(i) = x(i); 
    for (int i = 0; i < n; ++i) {
      y(i) = x(i) + 1e-8; 
      if (i > 0) y(i - 1) = x(i - 1); 
      grad(i) = (f(y) - f0) / (1e-8);   
    } 
    return(grad); 
  }
};



#endif // fmin_glennie_h



