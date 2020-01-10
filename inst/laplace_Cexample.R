# Running C_example.cpp to optimize a function in C++
library(RcppEigen)
library(Rcpp)
library(MASS)
library(mgcv)
# compile function
sourceCpp("laplace_Cexample.cpp")

# motorcycle crash acceleration data 
data(mcycle)

# motorcycle data
x <- mcycle$times
y <- mcycle$accel

# use mgcv
mod <- gam(y ~ s(x, bs = "cs"), method = "REML")

# get data structures 
gamdat <- gam(y ~ s(x, bs = "cs"), method = "REML", fit = FALSE)
X <- gamdat$X
S <- gamdat$S[[1]]
x0 <- c(log(50), -5, mean(y))
U <- X[,-1]
r <- y - mean(y)
u0 <- solve(t(U) %*% U, t(U) %*% r)

# Run optimization
res <- doF(x0, u0, y, X, S, maxit = 2, hessupdate = 0, verbose = TRUE)

# Ouputs
res




