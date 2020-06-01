## Using fmin on test functions
library(fmin)

# Rosenbrock Banana function, optimum is at (1, 1, ..., 1)
banana <- function(x) {
  n <- length(x)
  f <- sum(100 * (x[-1] - x[-n]^2)^2 + (1 - x[-n])^2)
  return(f)
}

opt <- fmin(banana, c(0, 0), verbose = TRUE, save = TRUE)
check_fmin(opt)
test <- nlm(banana, c(0, 0))

# Rosenbrock Banana function, optimum is at (0, 0, ..., 0)
rastrigin <- function(x) {
  n <- length(x)
  f <- 10 * n + sum(x^2 - 10 * cos(2 * pi * x))
  return(f)
}

opt <- fmin(rastrigin, c(-1, -1), verbose = TRUE)
test <- nlm(rastrigin, c(-1, -1))

# ackley function, minimum at (0, 0)
ackley <- function(x) {
  f <- -20 * exp(-0.2 * sqrt(0.5 * sum(x^2))) -
    exp(0.5 * sum(cos(2 * pi * x))) + exp(1) + 20
  return(f)
}

ack <- fmin(ackley, c(1, 1), save = TRUE, verbose = TRUE)
check_fmin(ack)
test <- nlm(ackley, c(1, 1), hessian = TRUE)

# sphere function, minimum at (0, ..., 0)
sphere <- function(x) {
  f <- sum(x^2)
  return(f)
}
sph <- fmin(sphere, c(1, 2, 3, 4), verbose = TRUE)

# Beale function, minimum at (3, 0.5)
beale <- function(x) {
  f <- (1.5 - x[1] + prod(x))^2 + (2.25 - x[1] + x[1] * x[2]^2)^2 +
    (2.625 - x[1] + x[1] * x[2]^3)^2
  return(f)
}
beal <- fmin(beale, c(0, 0), verbose = TRUE)

# Booth function, minimum at (1, 3)
booth <- function(x) {
  f <- (x[1] + 2 * x[2] - 7)^2 + (2 * x[1] + x[2] - 5)^2
  return(f)
}
bth <- fmin(booth, c(0, 0), verbose = TRUE)

# Levy function, minimum at (1, 1)
levy <- function(x) {
  f <- sin(3 * pi * x[1])^2 + (x[1] - 1)^2*(1 + sin(3 * pi * x[2])^2) +
    (x[2] - 1)^2 * (1 + sin(2*pi*x[2])^2)
  return(f)
}
lev <- fmin(levy, c(0, 0), verbose = TRUE)

# Three hump function, minimum at (0, 0)
hump <- function(x) {
  f <- 2 * x[1]^3 - 1.05 * x[1]^4 + x[1]^6 / 6 + x[1] * x[2] + x[2]^2
  return(f)
}

hmm <- fmin(hump, c(1, 1), verbose = TRUE)
test <- nlm(hump, c(1, 1))
hump(test$estimate)
hump(hmm$estimate)
