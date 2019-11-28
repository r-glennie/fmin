library(RcppEigen)
library(Rcpp)
sourceCpp("inst/test_c.cpp")

system.time(doF())


fn <- function(x) {sum(x^2)}
grad(fn, c(1, 2))
fmin(fn, c(1, 2), hessupdate = -1)


