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
# fmin: algorithm to minimize function using a combination of Newton-updates with
# finite-different approximant Hessians and BFGS Hessians. Step size is selected using 
# a cubic interpolation. 
#
#################################################################################

#' Check if matrix is positive definite
#'
#' @param X matrix
#' @param tol all eigenvalues less than tol in absolute value are rounded to zero
#'
#' @return TRUE if positive definite, FALSE otherwise
check_positive_definite <- function(X, tol = 1e-10) {
  E <- eigen(X, only.values = TRUE)
  if(any(E$values < tol)) {
    return(FALSE)
  } else {
    return(TRUE)
  }
}

#' Add epsilon to diagonal of matrix until it is positive definite
#'
#' @param X a square matrix
#' @param epsilon amount to add on each iteration
#' @param maxit maximum number of iterations to try
#' @param tol tolerance for check_positive_definite
#'
#' @return altered matrix
make_positive_definite <- function(X, epsilon = 1e-7, maxit = 100, tol = 1e-10) {
  Y <- X
  I <- epsilon * diag(ncol(Y))
  iter <- 0
  while(!check_positive_definite(Y, tol = tol)) {
    iter <- iter + 1
    Y <- Y + I
    if (iter > maxit) {
      warning("Failed to make matrix positive definite. Increase hessupdate.")
      break
    }
  }
  return(Y)
}

# Compute norm of a vector v
compute_norm <- function(v) {
  return(sqrt(sum(v^2)))
}

#' Check if stopping criterion satisified
#'
#' @param theta current parameters
#' @param g gradient vector
#' @param delta step vector
#' @param iter current iteration number
#' @param tol tolerance for tests
#' @param maxit maximum number of iterations
#' @param conv if TRUE return whether convergence criterion, not just stopping criteria are satisified
#'
#' @return TRUE if criteria satisified
check_stop <- function(theta, g, delta, iter, tol, maxit, conv = FALSE) {
  should_stop <- done <- FALSE
  th <- theta + 1e-10
  if (max(abs(g / th)) < tol) should_stop <- done <- TRUE
  if (max(abs(delta / th)) < tol) should_stop <- done <- TRUE
  if (iter >= maxit) should_stop <- TRUE
  if(conv) should_stop <- done
  return(should_stop)
}

#' Do a Newton update with a Cholesky factor for the Hessian
#'
#' @param R Cholesky factor (upper triangular) for the Hessian
#' @param g gradient vector
#'
#' @return Updated Cholesky
get_newton_step <- function(R, g) {
  y <- forwardsolve(t(R), -g)
  res <- backsolve(R, y)
  return(res)
}

#' Do a partial Newton step with line search selected step size
#'
#' @param alpha previous 2 step sizes  (or 1 if first halfstep)
#' @param theta current iterateo
#' @param f objective function
#' @param der directional derviative of f in direction of gradient g
#' @param fval f(theta) current function value
#' @param fvals previous 2 functional values (or 1 if first halfstep)
#' @param newton_step full Newton update step
#' @param halfsteps number of halfsteps taken so far
#' @param ... other arguments to f
#'
#' @return vector with two most recent step sizes (entries 1, 2) and two most recent function evaluations (3, 4)
get_halfstep <- function(alpha, theta, f, der, fval, fvals, newton_step, halfsteps, ...) {
  if (halfsteps == 0) {
    # quadratic fit
    alpha[2] <- -der / (2 * (fvals[2] - fvals[1] - der))
    fvals[2] <- f(theta + alpha[2] * newton_step, ...)
  } else {
    # cubic fit
    v <- c(fvals[2] - fval - der * alpha[2],
           fvals[1] - fval - der * alpha[1])
    a <- v[1] * alpha[1]^2 - v[2] * alpha[2]^2
    b <- -v[1] * alpha[1]^3 + v[2] * alpha[2]^3
    c <- alpha[1]^2 * alpha[2]^2 * (alpha[2] - alpha[1])
    a <- a / c
    b <- b / c
    alpha[1] <- alpha[2]
    fvals[1] <- fvals[2]
    alpha[2] <- (-b + sqrt(b^2 - 3 * a * der)) / (3 * a)
    fvals[2] <- f(theta + alpha[2] * newton_step, ...)
  }
  if (alpha[2] > 0.5 * alpha[1]) alpha[2] <- 0.5 * alpha[1]
  if (alpha[2] < alpha[1] * 0.1) alpha[2] <- 0.1 * alpha[1]
  return(c(alpha, fvals))
}

#' Update Cholesky factor of the Hessian by BFGS method
#'
#' @param R Cholesky factor
#' @param gdif change in gradients
#' @param delta step taken
#'
#' @return Updated Cholesky factor R
bfgs_update <- function(R, gdif, delta) {
  down <- gdif / sqrt(abs(1e-10 + as.numeric(t(delta) %*% gdif)))
  up <- gdif / sqrt(abs(1e-10 + as.numeric(t(gdif) %*% delta)))
  R <- cholup(cholup(R, up, up = TRUE), down, up = FALSE)
  return(R)
}

# OUTPUTS:
#  a list with elements:
#   - estimate: a vector of the optimal estimates
#   - value: value of the objective function at the optimum
#   - g: gradient vector at the optimum
#   - H: hessian at the optimum
#   - conv: is TRUE if convergence was satisfied, otherwise may not have converged on optimum
#   - niter: number of iterations taken
#' Minimize a function f using a Quasi-Newton method
#'
#' @param f function to be minimised which takes parameter vector as first argument
#' @param start vector of starting values
#' @param gfn (optional) gradient function of f, uses finite differencing otherwise
#' @param Hfn (optional) hessian function of f, uses finite differencing otherwise
#' @param maxit maximum number of iterations to try
#' @param tol tolerance for stopping criteria
#' @param hessupdate number of iterations for which to update approximate BFGS Hessian with finite-difference Hessian. If set to zero, then always use finite-difference. If negative then never use it.
#' @param maxhalfsteps maximum number of halfsteps to take when ensuring gradient descent
#' @param save if TRUE then parameter values, function values, and gradients are saved for every iteration
#' @param verbose if TRUE, function value and then parameter value are printed for each iteration
#' @param digits number of digits to print when verbose = TRUE
#' @param ... other named arguments to f
#'
#' @description Algorithm uses initial Hessian equal to identity matrix and then updates it using BFGS formula: it updates by Cholesky factor. Every hessupdate iterations, the approximate BFGS Hessian is replaced by the finite-difference approximate. A cubic line search is used to determined step size. Step direction is by Newton's method. Stopping criteria are relative gradient and relative step length.
#'
#' @return a list with elements:
#' \itemize{
#'   \item estimate - a vector of the optimal estimates
#'   \item value -  value of the objective function at the optimum
#'   \item g - gradient vector at the optimum
#'   \item H - hessian at the optimum
#'   \item conv - is TRUE if convergence was satisfied, otherwise may not have converged on optimum
#'   \item niter: number of iterations taken
#' }
#' @export
fmin <- function(f,
                 theta,
                 gfn = NULL,
                 Hfn = NULL,
                 maxit = 1000,
                 tol = 1e-7,
                 hessupdate = 10,
                 maxhalfsteps = 10,
                 save = FALSE,
                 verbose = FALSE,
                 digits = 4,
                 ...) {
  bfgs <- TRUE
  # if not gradient or Hessian then use numDeriv
  if (is.null(gfn)) {
    gfn <- function(theta, ...) {grad(f, theta, ...)}
  }
  if (is.null(Hfn)) {
    Hfn <- function(theta, ...) {hessian(f, theta, ...)}
  } else {
    hessupdate <- 0
    bfgs <- FALSE
  }
  # if asked to save then create storage
  if (save) {
    save_pars <- matrix(0, nr = length(theta), nc = maxit)
    save_fvals <- rep(0, maxit)
    save_gr <- matrix(0, nr = length(theta), nc = maxit)
  }
  # setup loop
  iter <- 0
  loop <- TRUE
  conv <- FALSE
  fval <- f(theta, ...)
  if (verbose) {
    digs <- paste0("%.", digits, "f")
  }
  I <- R <- diag(length(theta))
  g <- gfn(theta, ...)
  lastupdate <- 0
  if (hessupdate < 0) hessupdate <- maxit + 1
  while (loop) {
    iter <- iter + 1
    lastupdate <- lastupdate + 1
    # do a Hessian replacement of BFGS with finite-difference
    if (lastupdate > hessupdate) {
      lastupdate <- 0
      H <- Hfn(theta, ...)
      # if not positive definite, then perturb diagonal until it is
      if (!check_positive_definite(H)) H <- make_positive_definite(H)
      if (check_positive_definite(H)) R <- chol(H)
    }
    # solve for step direction
    newton_step <- delta <- get_newton_step(R, g)
    # find stepsize
    halfsteps <- 0
    gval <- 1e-4 * t(newton_step) %*% g
    der <- t(g) %*% newton_step
    alpha <- c(1, 0)
    fvals <- c(fval, f(theta + newton_step, ...))
    # do partial steps
    while (f(theta + delta, ...) > fval + gval  & halfsteps < maxhalfsteps) {
      hres <- get_halfstep(alpha, theta, f, der, fval, fvals, newton_step, halfsteps, ...)
      alpha <- hres[1:2]
      fvals <- hres[3:4]
      gval <- 10^(-4) * alpha[2] * der
      delta <- alpha[2] * newton_step
      halfsteps <- halfsteps + 1
    }
    # update theta
    theta <- theta + delta
    # update Hessian Cholesky factor
    fval <- f(theta, ...)
    gnew <- gfn(theta, ...)
    if (bfgs) {
      gdif <- gnew - g
      R <- bfgs_update(R, gdif, delta)
      g <- gnew
    }
    # print if asked
    if (verbose) cat(iter, "\t",
                     " ", sprintf(digs, signif(fval, 4)),
                     " | ", sprintf(digs, signif(theta, 4)),
                     "\t",
                     "\n")
    # check stopping criterion
    if (check_stop(theta, g, delta, iter, tol, maxit)) loop <- FALSE
    # save if asked
    if (save) {
      save_pars[,iter] <- theta
      save_fvals[iter] <- fval
      save_gr[,iter] <- g
    }
  }
  # check convergence
  if (check_stop(theta, g, delta, iter, tol, maxit, conv = TRUE)) {
    conv <- TRUE
  } else {
    conv <- FALSE
    warning("Failed to converge.")
  }
  res <- list(estimate = theta,
              value = fval,
              g = g,
              H = Hfn(theta, ...),
              conv = conv,
              niter = iter)
  if (save) res$save <- list(estimate = save_pars[,1:iter],
                             value = save_fvals[1:iter],
                             g = save_gr[,1:iter])
  return(res)
}

#' Check optimization from data stored during iterations in fmin
#'
#' @param opt output of fmin when fmin is run with save = TRUE argument
#'
#' @return four plots: norm of difference between parameters and final estimates
#'                     value of objective function over iterations
#'                     norm of gradient over iterations
#'                     norm of relative gradient over iterations
#' @export
check_fmin <- function(opt) {
  if (is.null(opt$save)) stop("opt must be output of fmin when run with save = TRUE")
  par(mfrow=c(2, 2))
  on.exit(par(mfrow=c(1,1)))
  par <- apply(opt$save$estimate - opt$estimate, 2, FUN = compute_norm)
  fval <- opt$save$value
  gval <- apply(opt$save$g, 2, FUN = compute_norm)
  relg <- apply(opt$save$g / opt$save$estimate, 2, FUN = compute_norm)
  iters <- 1:opt$niter
  plot(iters, par, xlab = "Iterations", ylab = "", main = "Parameter difference from final estimate", bty = "l", type = "l")
  plot(iters, fval, xlab = "Iterations", ylab = "", main = "Objective function value", bty = "l", type = "l")
  plot(iters, gval, xlab = "Iterations", ylab = "", main = "Norm of Gradient", bty = "l", type = "l")
  plot(iters, relg, xlab = "Iterations", ylab = "", main = "Norm of Relative Gradient", bty = "l", type = "l")
  invisible(opt)
}


