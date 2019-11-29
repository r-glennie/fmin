library(RcppEigen)
library(Rcpp)
sourceCpp("inst/test_c.cpp")

res <- system.time(doF(c(0, 0), hessupdate = 10, verbose = TRUE))
system.time(fmin(banana, c(2, 2), hessupdate = 0, verbose = TRUE))
hessian(banana, c(1, 1))

system.time(nlm(banana, c(2, 2)))

fn <- function(x) {sum(x^2)}
grad(fn, c(1, 2))
fmin(banana, c(1, 2), hessupdate = 0)

y <- c(1,1)
h <- 1e-8

f <- banana
g1 <- f(y + c(2 * h, 0)) - 2 * f(y) + f(y + c(-2*h, 0)) / h^2
g2 <- f(y + c(h, 0)) - 2 * f(y) + f(y + c(-h, 0)) / h^2

(2^2 * g2 - g1) / 3

d <- (g1 / 12 - 2 * g2 / 3 + g3 * 2 / 3  - g2 /12 ) / h

H <- hessian(banana, y)
H
