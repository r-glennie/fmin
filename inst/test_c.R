library(RcppEigen)
library(Rcpp)
sourceCpp("inst/test_c.cpp")

system.time(doF())

system.time(nlm(banana, c(1, 2)))

fn <- function(x) {sum(x^2)}
grad(fn, c(1, 2))
fmin(banana, c(1, 2), hessupdate = 50)


