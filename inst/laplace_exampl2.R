## Optimize marginal function

# simulate data
n <- 100
x <- runif(n)
y <- cos(2 * pi * x) + rnorm(n, 0, 0.05)

plot(x, y)

# use mgcv
library(mgcv)
mod <- gam(y ~ s(x, bs = "cs"), method = "REML")
plot(mod)
coef(mod)

f <- function(u, theta, dat) {
  ind <- 1:length(theta)
  ind <- ind[-(1:dat$npar)]
  nu <- dat$X %*% c(theta[ind], u)
  llk <- sum(dnorm(dat$y, nu, exp(theta[1]), log = TRUE))
  llk <- llk + theta[2] * ncol(dat$S) / 2 - exp(theta[2]) * t(u) %*% dat$S %*% u / 2
  return(-as.numeric(llk))

}

gamdat <- gam(y ~ s(x, bs = "cs"), method = "REML", fit = FALSE)
dat <- list(y = y, X = gamdat$X, S = gamdat$S[[1]])

u0 <- rep(0, ncol(dat$S))
theta0 <- c(0, 0, mean(dat$y))
dat$npar <- 2

m <- fminlaplace(f, u0, theta0, verbose = TRUE, hessupdate = 0, dat = dat)

pred0 <- predict(mod)
pred1 <- dat$X %*% c(m$estimate[3], m$inner$estimate)

plot(pred0, pred1)
abline(a = 0, b = 1)
