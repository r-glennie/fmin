###############################################################################
#
# fmin: algorithm to minimize function using line search Quasi-Newton method
#       as described in Nocedal and Wright (2006). Line Search uses bracketing
#       and quadratic interpolation to satisfy Wolfe conditions. Inverse
#       hessian is updated by BFGS.

###############################################################################

# dot product of vectors
dot <- function(v, w) {
  return(sum(v * w))
}

#' Check if stopping criterion satisified
#'
#' @param g gradient vector
#' @param delta step vector
#' @param iter current iteration number
#' @param tol tolerance for tests
#' @param maxit maximum number of iterations
#' @param conv if TRUE return whether convergence criterion, not just stopping
#'             criteria are satisified
#'
#' @return TRUE if criteria satisified
check_stop <- function(g,
                       delta,
                       iter,
                       tol,
                       maxit,
                       conv = FALSE) {
  should_stop <- done <- FALSE
  if (max(abs(g)) < tol) should_stop <- done <- TRUE
  if (max(abs(delta)) < tol) should_stop <- done <- TRUE
  if (iter >= maxit) should_stop <- TRUE
  if(conv) should_stop <- done
  return(should_stop)
}

#' Do a Newton update with inverse Hessian
#'
#' @param H Inverse Hessian matrix
#' @param g gradient vector
#'
#' @return Newton step vector
get_newton_step <- function(H, g) {
  res <- -H %*% g
  return(res)
}

#' Backtrack to find a new candidate step size
#'
#' @param alpha previous 2 step sizes
#' @param fvals previous 2 functional values for alpha step sizes
#'              (or 1 if first substep)
#' @param minstep minimum relative step size change
#'
#' @description See p58 of Nocedal and Wright (2006).
#'
#' @return proposed step size
backtrack <- function(alpha,
                         fvals,
                         gval,
                         minstep = 0.1) {
  # quadratic approximation
  alpdiff <- alpha[2] - alpha[1]
  a <- -gval * alpdiff^2
  b <- fvals[2] - fvals[1] - alpdiff * gval
  incr <- a / (2 * b)
  newalp <- alpha[1] + incr
  # prevent steps that are too similar
  m <- min(alpha)
  M <- max(alpha)
  r <- M - m
  if ((newalp - m) / r < minstep) newalp <- m + minstep * r
  if ((M - newalp) / r < minstep) newalp <- M - minstep * r
  return(newalp)
}

# check sufficient decrease condition (first Wolfe condition)
sufficient_decrease <- function(fnew, fval, alpha, der, c1 = 1e-4) {
  return(fnew <= fval + c1 * alpha * der)
}

# check curvature condition (second Wolfe condition)
curvature_condition <- function(dernew, der, c2 = 0.9) {
  return(abs(dernew) <= -c2 * der)
}

#' Perform line search in direction of newton step
#' @param theta current function values
#' @param newton_step full newton step
#' @param fval current function value
#' @param der directional derivative of f in direction of newton_step
#' @param f objective function
#' @param gfn gradient function
#' @param maxsubsteps maximum substeps to perform
#' @param stepmax maximum step size allowed
linesearch <- function(theta,
                       newton_step,
                       fval,
                       der,
                       f,
                       gfn,
                       maxsubsteps,
                       stepmax,
                       funit,
                       units) {
  # begin with full Newton step size of 1
  alpha <- c(0, 1)
  # iniitialise function at proposed Newton step
  fvals <- c(fval, f(theta + alpha[2] * newton_step))
  gval <- der
  substeps <- 1
  repeat {
    suff <- sufficient_decrease(fvals[2], fval, alpha[2], der)
    # if no sufficient decrease anymore, then zoom
    if (!suff | (substeps > 1 & fvals[2] > fvals[1])) {
      alpha[2] <- zoom(alpha,
                       theta,
                       f,
                       gfn,
                       der,
                       fval,
                       fvals,
                       gval,
                       newton_step,
                       maxsubsteps)
      break
    }
    newg <- gfn(theta + alpha[2] * newton_step)
    dernew <- dot(newg, newton_step)
    # if sufficient decrease and curvature good, stop
    if (curvature_condition(dernew, der)) {
      break
    }
    # if sufficient decrease and positive curvature, zoom
    if (dernew >= 0) {
      alpha[2] <- zoom(rev(alpha),
                       theta,
                       f,
                       gfn,
                       der,
                       fval,
                       rev(fvals),
                       gval,
                       newton_step,
                       maxsubsteps)
      break
    }
    # sufficient decrease but poor curvature, extend search
    alpha[1] <- alpha[2]
    fvals[1] <- fvals[2]
    gval <- dernew 
    alpha[2] <- 2 * alpha[2]
    fvals[2] <- f(theta + alpha[2] * newton_step)
    # stop if maximum number of substeps taken
    if (substeps > maxsubsteps | alpha[2] > stepmax) {
      break
    }
    substeps <- substeps + 1
  }
  return(alpha[2])
}

#' Zoom in on step size given interval where Wolfe conditions can be satisfied
#' @param alpha vector of 2 step sizes, boundary of interval
#' @param theta current parameter values
#' @param f objective function
#' @param gfn gradient function
#' @param der direction derivative of f in Newton direction
#' @param fval current function value
#' @param fvals function values at boundary step sizes
#' @param newton_step full Newton step direction
#' @param maxsubsteps maximum number of substeps allowed
zoom <- function(alpha,
                 theta,
                 f,
                 gfn,
                 der,
                 fval,
                 fvals,
                 gval,
                 newton_step,
                 maxsubsteps) {
  substeps <- 1
  oldf <- fval
  repeat {
    # interpolate within interval to new step size
    alp <- backtrack(alpha, fvals, gval)
    # compute function value at new step
    newf <- f(theta + alp * newton_step)
    # if no sufficient decrease make this new step the upper bound
    if (!sufficient_decrease(newf, fval, alp, der) | newf > oldf) {
      alpha[2] <- alp
      fvals[2] <- newf
      oldf <- newf
    } else {
      # compute gradient and direction derivative at new step
      newg <- gfn(theta + alp * newton_step)
      newder <- dot(newg, newton_step)
      # curvature condition satisfied then accept step
      if (curvature_condition(newder, der)) {
        break
      }
      # make alpha[2] the step size with best curvature
      if (newder * (alpha[2] - alpha[1]) >= 0) {
        alpha[2] <- alpha[1]
        fvals[2] <- fvals[1]
      }
      # make new step the best found so far that has sufficient decrease
      alpha[1] <- alp
      fvals[1] <- newf
      gval <-  newder
    }
    substeps <- substeps + 1
    if (substeps > maxsubsteps) {
      break
    }
  }
  return(alp)
}

#' Update inverse Hessian by BFGS method
#'
#' @param H inverse Hessian
#' @param gdif change in gradients
#' @param delta step taken
#'
#' @return Updated inverse Hessian H
bfgs_update <- function(H, gdif, delta) {
  r <- 1 / dot(gdif, delta)
  I <- diag(nrow(H))
  newH <- (I - r * delta %*% t(gdif)) %*% H %*%
          (I - r * gdif %*% t(delta)) + r * delta %*% t(delta)
  return(newH)
}

#' Minimize a function f using a Quasi-Newton method
#'
#' @param obj function to be minimised which takes parameter vector as
#'          first argument
#' @param start vector of starting values
#' @param gobj (optional) gradient function of f, uses finite differencing
#'             otherwise
#' @param funit scaling for function values, try to pick this so that f 
#'              values vary between -1 and 1
#' @param units a vector of same length as start giving coordinate units, 
#'              try to pick units such that values of parameter in that
#'              coordinate direction vary between -1 and 1
#' @param maxit maximum number of iterations to try
#' @param tol tolerance for stopping criteria
#' @param stepmax maximum relative step length, default of 1 means no steps
#'                longer than full Newton step 
#' @param maxsubsteps maximum number of substeps in line search
#' @param save if TRUE then parameter values, function values, and gradients
#'             are saved for every iteration
#' @param verbose if TRUE, function value and then parameter value are printed
#'                for each iteration
#' @param digits number of digits to print when verbose = TRUE
#' @param ... other named arguments to f and gobj (if given)
#'
#' @description Performs a quasi-Newton line search optimization with BFGS
#'              updates of the inverse hessian. See Nocedal and Wright
#'              (2006), Chapter 3 for details.
#'
#' @return a list with elements:
#' \itemize{
#'   \item estimate - a vector of the optimal estimates
#'   \item value -  value of the objective function at the optimum
#'   \item g - gradient vector at the optimum
#'   \item H - hessian at the optimum computed by finite difference
#'   \item conv - is TRUE if convergence was satisfied, otherwise may not have
#'                 converged on optimum
#'   \item niter: number of iterations taken
#' }
#' @export
fmin <- function(obj,
                 theta,
                 gobj = NULL,
                 funit = 1,
                 units = NULL,
                 maxit = 200,
                 tol = 1e-10,
                 stepmax = 1,
                 maxsubsteps = 10,
                 save = FALSE,
                 verbose = FALSE,
                 digits = 4,
                 ...) {
  # default coordinate units are 1
  if (is.null(units)) units <- rep(1, length(theta))
  # define scaled objective function
  f <- function(theta) {
    obj(units * theta, ...) / funit
  }
  # if not gradient supplied then use numDeriv
  if (is.null(gobj)) {
    gfn <- function(theta) {
      grad(f, theta, ...)
    }
  } else {
    gfn <- function(theta) {
      gobj(units * theta, ...) / funit
    }
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
  fval <- f(theta)
  if (verbose) {
    digs <- paste0("%.", digits, "f")
  }
  # initial inverse Hessian is identity
  H <- diag(length(theta))
  # initial gradient
  g <- gfn(theta)
  while (loop) {
    iter <- iter + 1
    # compute step direction
    newton_step <- delta <- get_newton_step(H, g)
    der <- dot(g, newton_step)
    # find stepsize
    step_size <- linesearch(theta,
                            newton_step,
                            fval,
                            der,
                            f,
                            gfn,
                            maxsubsteps,
                            stepmax)
    # compute step 
    delta <- step_size * newton_step
    # update theta
    theta <- theta + delta
    # update Hessian
    fval <- f(theta)
    gnew <- gfn(theta)
    gdif <- gnew - g
    # if first iterate, replace identity Hessian with better approximation
    # before BFGS update
    if (iter == 1) H <- dot(gdif, delta) / dot(gdif, gdif) * H
    H <- bfgs_update(H, gdif, delta)
    g <- gnew
    # print if asked
    if (verbose) cat(iter, "\t",
                     " ", sprintf(digs, signif(fval * funit, 4)),
                     " | ", sprintf(digs, signif(theta * units, 4)),
                     "\t",
                     "\n")
    # check stopping criterion
    if (check_stop(g, delta, iter, tol, maxit)) loop <- FALSE
    # save if asked
    if (save) {
      save_pars[, iter] <- theta * units
      save_fvals[iter] <- fval * funit
      save_gr[, iter] <- g * funit
    }
  }
  # check convergence
  if (check_stop(g, delta, iter, tol, maxit, conv = TRUE)) {
    conv <- TRUE
  } else {
    conv <- FALSE
    warning("Failed to converge.")
  }
  res <- list(estimate = as.vector(theta * units),
              value = fval * funit,
              g = g * funit,
              H = hessian(obj, theta * units, ...),
              conv = conv,
              niter = iter)
  if (save) res$save <- list(estimate = save_pars[, 1:iter],
                             value = save_fvals[1:iter],
                             g = save_gr[, 1:iter])
  return(res)
}

#' Check optimization from data stored during iterations in fmin
#'
#' @param opt output of fmin when fmin is run with save = TRUE argument
#'
#' @return four plots: norm of difference between parameters and final estimates
#'                     value of objective function over iterations
#'                     norm of gradient over iterations
#'                     maximum absolute gradient component over iterations 
#' @export
check_fmin <- function(opt) {
  if (is.null(opt$save)) stop("opt must be output of fmin when run with
                              save = TRUE")
  par(mfrow = c(2, 2))
  on.exit(par(mfrow = c(1 ,1)))
  par <- apply(opt$save$estimate - opt$estimate, 2, FUN = function(x){sqrt(sum(x^2))})
  fval <- opt$save$value
  normg <- apply(opt$save$g, 2, FUN = function(x) {sqrt(sum(x^2))})
  maxg <- apply(opt$save$g, 2, FUN = function(x){max(abs(x))})
  iters <- 1:opt$niter
  plot(iters,
       par,
       xlab = "Iterations",
       ylab = "",
       main = "Parameter difference from final estimate",
       bty = "l",
       type = "l")
  plot(iters,
       fval,
       xlab = "Iterations",
       ylab = "",
       main = "Objective function value",
       bty = "l",
       type = "l")
  plot(iters,
       normg,
       xlab = "Iterations",
       ylab = "",
       main = "Gradient",
       bty = "l",
       type = "l")
  plot(iters,
       maxg,
       xlab = "Iterations",
       ylab = "",
       main = "Maximum Absolute Gradient",
       bty = "l",
       type = "l")
  invisible(opt)
}
