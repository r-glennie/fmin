## Optimize marginal function
library(MASS)
data(mcycle)

# motorcycle data
x <- mcycle$times
y <- mcycle$accel

# use mgcv
library(mgcv)
mod <- gam(y ~ s(x, bs = "cs"), method = "REML")
plot(mod)
coef(mod)

# define full likelihood
f <- function(u, par, dat) {
  ind <- 1:length(par)
  ind <- ind[-(1:dat$npar)]
  nu <- dat$X %*% c(par[ind], u)
  llk <- sum(dnorm(dat$y, nu, exp(par[1]), log = TRUE))
  llk <- llk + par[2] * ncol(dat$S) / 2 - exp(par[2]) * t(u) %*% dat$S %*% u / 2
  return(-as.numeric(llk))
}

fmarg <- function(par, u0, dat) {
  opt <- fmin(f, u0, par = par, dat = dat)
  #Hldet <- sum(log(diag(chol(opt$H))))
  Hldet <- determinant(opt$H, logarithm = TRUE)
  l <- -opt$value - Hldet$modulus / 2
  nl <- -l
  attributes(nl)$opt <- opt
  return(nl)
}

gamdat <- gam(y ~ s(x, bs = "cs"), fit = FALSE)
dat <- list(y = y, X = gamdat$X, S = gamdat$S[[1]])

u0 <- rep(0, ncol(dat$S))
theta0 <- c(log(50), 0, mean(dat$y))
dat$npar <- 2

f(u0, theta0, dat)

fmarg(theta0, u0, dat)

m <- fmin(fmarg, theta0, u0 = u0, dat = dat, verbose = TRUE, hessupdate = 0)

pred0 <- predict(mod)
pred1 <- dat$X %*% c(m$estimate[3], attributes(m$value)$opt$estimate)
plot(pred0, pred1)
abline(a = 0, b = 1)
cbind(pred0, pred1)

attributes(m$value)$opt$estimate

