#ifndef REGRESSION_H
#define REGRESSION_H

void __fit(
    const float* X,
    const float* y,
    int max_iterations,
    float lr,
    float tol,
    float* theta,
    int n,
    int k
);

void __predict(
    const float* X,
    const float* theta, 
    float* prediction_ptr,
    int n,
    int k
);

#endif