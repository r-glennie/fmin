% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/fmin.R
\name{check_stop}
\alias{check_stop}
\title{Check if stopping criterion satisified}
\usage{
check_stop(theta, fval, g, delta, iter, tol, maxit, conv = FALSE)
}
\arguments{
\item{theta}{current parameters}

\item{fval}{current function value}

\item{g}{gradient vector}

\item{delta}{step vector}

\item{iter}{current iteration number}

\item{tol}{tolerance for tests}

\item{maxit}{maximum number of iterations}

\item{conv}{if TRUE return whether convergence criterion, not just stopping
criteria are satisified}
}
\value{
TRUE if criteria satisified
}
\description{
Check if stopping criterion satisified
}
