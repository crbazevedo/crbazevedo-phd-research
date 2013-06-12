#include "../headers/mvtnorm.h"

#include <cstdlib>
#include <cstdio>

/**
 * @return <0 if anything goes wrong.
 */
double pmvnorm(int* n,
               int* nu,
               double* lower,
               double* upper,
               int* infin,
               double* correl,
               double* delta, // non-central parameter
               int* maxpts,    // param
               double* abseps, // param
               double* releps, // param
               double* error,  // estimated abs. error. with 99% confidence interval
               double* value,     // results store here.
               int* inform)    // inform message goes here
{
  mvtdst_ (n, nu,
           lower, upper, infin, correl, delta,
           maxpts, abseps, releps,
           error, value, inform);
  //printf ("error = %g, value = %g, inform = %d\n", *error, *value, *inform);
  
  switch (*inform) {
    case 0:
      return *value;
    case 1:
    case 2:
    case 3:
      return -1.0;
  };

  return *value;
};

/**
 * @return CDF of multivariate normal P ( X < bound ) where X ~ MVN(0, correlationMatrix)
 */
double pmvnorm_P(int n,
                 double* bound,
                 double* correlationMatrix, // (2,1), (3,1), (3,2) .....
                 double* error)
{
  int nu_ = 0;
  int maxpts_ = 25000;
  double abseps_ = 1e-6;
  double releps_ = 0;


  double* lower = new double[n];
  int* infin = new int[n];
  double* delta = new double[n];

  int i = 0;
  for (i = 0; i < n; ++i) {
    infin[i] = 0; // (-inf, bound]
    lower[i] = 0.0;
    delta[i] = 0.0;
  }

  // return values
  double value_ = 0.0;
  int inform_ = 0.0;

  double ret = pmvnorm(&n, &nu_, lower, bound, infin, correlationMatrix, delta, &maxpts_, &abseps_, &releps_, error, &value_, &inform_);
  delete[] (lower);
  delete[] (infin);
  delete[] (delta);

  return ret;
}

/**
 * @return (1 - CDF) of multivariate normal P ( X > bound ) where X ~ MVN(0, correlationMatrix)
 */
double pmvnorm_Q(int n,
                 double* bound,
                 double* correlationMatrix, // (2,1), (3,1), (3,2) .....
                 double* error)
{
  int nu_ = 0;
  int maxpts_ = 25000;
  double abseps_ = 1e-6;
  double releps_ = 0;


  double* upper = new double[n];
  int* infin = new int[n];
  double* delta = new double[n];

  int i = 0;
  for (i = 0; i < n; ++i) {
    infin[i] = 1; // (-inf, bound]
    upper[i] = 0.0;
    delta[i] = 0.0;
  }

  // return values
  double value_ = 0.0;
  int inform_ = 0.0;

  double ret = pmvnorm(&n, &nu_, bound, upper, infin, correlationMatrix, delta, &maxpts_, &abseps_, &releps_, error, &value_, &inform_);
  delete[] (upper);
  delete[] (infin);
  delete[] (delta);

  return ret;
}
