#include "regression.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

void __predict(
    const float* X,
    const float* theta, 
    float* prediction_ptr,
    int n,
    int k
)
{
    const float alpha = 1.0f; // Required for sgemv 
    const float beta = 0.0f;
    
    size_t X_size = n * k * sizeof(float);
    size_t y_size = n * sizeof(float);
    size_t theta_size = k * sizeof(float);
    
    float *cuda_X, *cuda_theta, *cuda_prediction;

    cudaMalloc(
        &cuda_X, // ptr
        X_size   // size
    );

    cudaMalloc(&cuda_theta, theta_size);
    cudaMalloc(&cuda_prediction, y_size);
    
    cudaMemcpy(
        cuda_X,                   // dst
        X,                        // src
        X_size,                   // size
        cudaMemcpyHostToDevice    // kind
    );

    cudaMemcpy(
        cuda_theta,
        theta,
        theta_size,
        cudaMemcpyHostToDevice
    );
    
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    
    cublasSgemv(
        cublas_handle,      
        CUBLAS_OP_T,        // transpose 1 x k to k x 1
        k, n,               // shape                         
        &alpha,             // alpha=1
        cuda_X,             // A=X
        k,                  // lda=features
        cuda_theta,         // x=theta
        1,                  // incx=1
        &beta,              // beta=0
        cuda_prediction,    // y=prediction
        1                   // incy=1
    );                   
    
    cudaMemcpy(
        prediction_ptr,        // Return prediction value back to the host   
        cuda_prediction,       
        y_size,                 
        cudaMemcpyDeviceToHost
    );
    
    cublasDestroy(cublas_handle); 
    cudaFree(cuda_X);
    cudaFree(cuda_theta);
    cudaFree(cuda_prediction);
}