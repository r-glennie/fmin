# Copyright (c) 2019 Richard Glennie
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
#
# fminlaplace: algorithm to minimise the integral of function where the integration
#  is with respect to some of its arguments. The integral is approximated by the
#  Laplace approximation.
#
#
#################################################################################

#' Minimize marginalised function using Laplace approximation
#'
#' @param f function whose marginal is to be minimized. Function must have form f(nu, theta, ...) where
#'          nu are the parameters to be integrated out (marginalised), theta are the parameter to optimize,
#'          and ... are any other arguments to f.
#' @param nu starting values for the marginalized parameters
#' @param theta starting values for the parameters to be optimized
#' @param maxit maximum number of iterations
#' @param tol tolerance for stopping rules
#' @param hessupdate number of iterations for which to update approximate BFGS Hessian with finite-difference Hessian. If set to zero, then always use finite-difference. If negative then never use it.
#' @param maxhalfsteps maximum number of halfsteps to take when ensuring gradient descent
#' @param save if TRUE then parameter values, function values, and gradients are saved for every iteration
#' @param verbose if TRUE, function value and then parameter value are printed for each iteration
#' @param digits number of digits to print when verbose = TRUE
#' @param ... other named arguments to f
#'
#' @return a list with elements:
#' \itemize{
#'   \item estimate - a vector of the optimal estimates
#'   \item value -  value of the objective function at the optimum
#'   \item g - gradient vector at the optimum
#'   \item H - hessian at the optimum
#'   \item conv - is TRUE if convergence was satisfied, otherwise may not have converged on optimum
#'   \item niter: number of iterations taken
#'   \item inner: the output of fmin for the final inner optimization over nu, so inner$estimate are the
#'         optimal values for f at the optimal marginal parameters theta.
#' }
#' @export
fminlaplace <- function(f,
                 nu,
                 theta,
                 maxit = 1000,
                 tol = 1e-7,
                 hessupdate = 10,
                 maxhalfsteps = 10,
                 save = FALSE,
                 verbose = FALSE,
                 digits = 4,
                 ...) {
  # set starting value for to-be-marginalised variables
  nu0 <- nu
  inneropt <- NULL
  # define Laplace-approximate marginal pseudo-function
  fmarg <- function(theta, nu0, est = FALSE, ...) {
    # create subf
    subf <- function(nu) {return(f(nu, theta, ...))}
    # inner optimization is a purely Newton update with variable step-size (no BFGS)
    inneropt <- fmin(subf, nu0, hessupdate = 0)
    # log-determinant of the Hessian
    Hldet <- determinant(inneropt$H, log = TRUE)
    val <- inneropt$value + Hldet$modulus / 2
    # set new starting values
    nu0 <- inneropt$estimate
    if (est) return(inneropt)
    return(val)
  }
  # do outer optimization (optimizing the marginal function)
  outeropt <- fmin(fmarg,
                   theta,
                   nu0 = nu0,
                   maxit = maxit,
                   tol = tol,
                   hessupdate = hessupdate,
                   maxhalfsteps = maxhalfsteps,
                   save = save,
                   verbose = verbose,
                   digits = digits,
                   ...)
  outeropt$inner <- fmarg(outeropt$estimate, nu0, est = TRUE, ...)
  return(outeropt)
}


fullCovariance <- function(f, opt, ...) {

  # define Laplace-approximate marginal pseudo-function
  fmarg <- function(theta, nu0, est = FALSE, ...) {
    # create subf
    subf <- function(nu) {return(f(nu, theta, ...))}
    # inner optimization is a purely Newton update with variable step-size (no BFGS)
    inneropt <- fmin(subf, nu0, hessupdate = 0)
    # log-determinant of the Hessian
    Hldet <- determinant(inneropt$H, logarithm = TRUE)
    val <- inneropt$value + Hldet / 2
    # set new starting values
    nu0 <- inneropt$estimate
    if (est) return(inneropt)
    return(val)
  }

  # compute (nu, theta) at Laplace mode
  parfn <- function(theta, nu0) {
    nuhat <- fmarg(theta, nu0, est = TRUE, ...)$estimate
    return(c(nuhat, theta))
  }
  # jacobian with respect to (nu, theta)
  J <- jacobian(parfn, opt$estimate, nu0 = opt$inner$estimate)
  # variance of nu conditional on theta
  nfix <- length(opt$estimate)
  nranef <- length(opt$inner$estimate)
  npar <- nfix + nranef
  Vnu <- solve(opt$inner$H)
  V <- matrix(0, nr = npar, nc = npar)
  V[1:nranef, 1:nranef] <- Vnu
  Vtheta <- solve(opt$H)
  V <- V + J %*% Vtheta %*% t(J)
  return(V)
}





