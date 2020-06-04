# Running C_example.cpp to optimize a function in C++
library(RcppEigen)
library(Rcpp)
# compile function
sourceCpp("C_example.cpp")

# Run optimization
res <- doF(c(0, 0), verbose = TRUE)

# Ouputs
res
