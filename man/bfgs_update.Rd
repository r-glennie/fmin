% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/fmin.R
\name{bfgs_update}
\alias{bfgs_update}
\title{Update inverse Hessian by BFGS method}
\usage{
bfgs_update(H, gdif, delta)
}
\arguments{
\item{H}{inverse Hessian}

\item{gdif}{change in gradients}

\item{delta}{step taken}
}
\value{
Updated inverse Hessian H
}
\description{
Update inverse Hessian by BFGS method
}
