/* ===================================================
    __fit.cu
 
    Trains the Linear Regression model using a 
    vectorized approach to the Gradient Descent
 
    Model Specification: \widehat{y} = X @ theta
 
    Training methodology: \epsilon = \frac{1}{2n}|| X*theta-y ||^2
 
    --------------------------------------------------
    Diagram:
 
    CPU:
    X_host, y_host, theta_host (allocated space)
 
        | copy
        V
    GPU:
    X, y, theta
 
    Training loop:
    Predict -> cublasSgemv()
    Residuals -> calculateResiduals()
    Gradient -> cublasSgemv()
    Update -> updateWeight()
 
        | copy
        V
    CPU:
    theta at convergence 

       | send to python
       V

    model = LinearRegression()
    model.fit(X,y)
    model.theta = [...]

   =================================================== */ 

#include "regression.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

/* ===================================================
   calculateResiduals(y_pred, *y, *error, n)

   Compute one sample per thread with:

   \epsilon = sum i=0 ... n-1 (\widehat{y} - y)

   =================================================== */ 

__global__ void calculateResiduals(
    const float* y_pred, const float* y, 
    float* error, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n)
    {
        error[idx] = y_pred[idx] - y[idx];
    }
}

/* ===================================================
   updateWeight(*theta, *gradient, lr, k)

   Efficient vectorized operation designed for GPU.

   Called within the gradient descent loop to update the
   model weights and bias. 

   GPU can load 128 bits or 4 floats in a single instruction
   so for all k % 4 == 0, the processing is vectorized,
   else the remaining elements are scalar processed.

   =================================================== */ 

__global__ void updateWeight(
    float* theta, const float* gradient, 
    float lr, int k) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;

    if (vec_idx + 3 < k)        // k % 4 == 0
    {
        float4* theta_vec = (float4*)theta;
        const float4* grad_vec = (const float4*)gradient;
        
        float4 t = theta_vec[idx];
        float4 g = grad_vec[idx];
        
        t.x -= lr * g.x;
        t.y -= lr * g.y;
        t.z -= lr * g.z;
        t.w -= lr * g.w;
        
        theta_vec[idx] = t;
    }
    else if (vec_idx < k)       // Fallback
    {
        for (int i = vec_idx; i < k && i < vec_idx + 4; i++)
        {
            theta[i] -= lr * gradient[i];
        }
    }
}

/* ===================================================
   reduceLoss(*error, *loss, n)

   Each thread computes the SSR/SSE for multiple samples

   =================================================== */ 

__global__ void reduceLoss(
    const float* error,
    float* loss, int n) {

    __shared__ float shared_loss[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    float local_loss = 0.0f;
    
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        float err = error[i];
        local_loss += err * err; // Sum of squared errors
    }
    
    shared_loss[tid] = local_loss;
    __syncthreads();
    
    // Optimized reduction
    for (int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            shared_loss[tid] += shared_loss[tid + s];
        }
        __syncthreads();
    }
    
    // Warp-level reduction (no sync needed)
    if (tid < 32)
    {
        volatile float* smem = shared_loss;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8) smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4) smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2) smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0)
    {
        atomicAdd(loss, shared_loss[0]);
    }
}

/* ===================================================
   fit(*X, *y, m_iters, lr, tol, *theta, n, k)

   =================================================== */ 
   
void __fit(
    const float* X_host,
    const float* y_host,
    int max_iterations,
    float lr,
    float tol,
    float* theta_host,
    int n, int k) {

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    const int loss_check_interval = 200; 
    
    size_t X_size = n * k * sizeof(float);
    size_t y_size = n * sizeof(float);
    size_t theta_size = k * sizeof(float);
    
    float *d_X, *d_y, *d_theta, *d_gradient, *d_y_pred, *d_error, *d_loss;

    cudaMalloc(&d_X, X_size);
    cudaMalloc(&d_y, y_size);
    cudaMalloc(&d_theta, theta_size);
    cudaMalloc(&d_gradient, theta_size);
    cudaMalloc(&d_y_pred, y_size);
    cudaMalloc(&d_error, y_size);
    cudaMalloc(&d_loss, sizeof(float));
    
    cudaMemcpy(d_X, X_host, X_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y_host, y_size, cudaMemcpyHostToDevice);
    
    cudaMemset(d_theta, 0, theta_size); // Initialize theta
    
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    
    int block_size = 256;
    int grid_size_samples = (n + block_size - 1) / block_size;
    int grid_size_theta = (k + block_size * 4 - 1) / (block_size * 4);

    float prev_loss = 1e10f;

    cudaStream_t stream1, stream2;        // Separate streams
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    

    printf("Fitting Linear Regression Gradient with %s\n", prop.name);
    printf("Max Iterations: %d | Learning Rate: %.4f | Tolerance: %.8f\n\n", max_iterations, lr, tol);

    for (int iter = 0; iter < max_iterations; iter++)
    {
        const float alpha = 1.0f;       // predict
        const float beta = 0.0f;        // for gradient and predict
        const float scale = 1.0f / n;   // for gradient
        
        // Predict on each iteration with CUBLAS_OP_T to transpose the X matrix.
        cublasSgemv(
           cublas_handle,
           CUBLAS_OP_T,
           k, n,
           &alpha,
           d_X,
           k,
           d_theta, // This will start at a vector of 0s
           1,
           &beta,
           d_y_pred, // Write into
           1
        );
        
        /*
        Check if converged within tol each 200 iterations
        */
        if (iter % loss_check_interval == 0) 
        {
            
            calculateResiduals<<<grid_size_samples, block_size, 0, stream1>>>
            (d_y_pred, d_y, d_error, n);
            
            cudaMemsetAsync(d_loss, 0, sizeof(float), stream1);

            reduceLoss<<<grid_size_samples, block_size, 0, stream1>>>
            (d_error, d_loss, n);
            
            float current_loss;

            cudaMemcpyAsync(
                &current_loss,
                d_loss,
                sizeof(float), 
                cudaMemcpyDeviceToHost,
                stream1
            );

            cudaStreamSynchronize(stream1);
            
            current_loss /= (2.0f * n);  // MSE
            
            if (iter > 0 && fabs(prev_loss - current_loss) < tol)
            {
                printf("\nConverged within tolerance at iteration %d with loss %.6f\n", iter, current_loss);
                break;
            }

            prev_loss = current_loss;
            printf("Iteration %d | MSE: %.6f\n", iter, current_loss);
        }
        if (iter % loss_check_interval != 0)  // Ensure residuals have been calculated 
        {
            calculateResiduals<<<grid_size_samples, block_size>>>
            (d_y_pred, d_y, d_error, n);
        }
        
        // (gradient = X^T * error / m)
        cublasSgemv(
            cublas_handle,
            CUBLAS_OP_N,
            k, n,
            &scale,
            d_X,
            k,
            d_error,
            1,
            &beta,
            d_gradient, // Write into
            1
        );
        
        updateWeight<<<grid_size_theta, block_size>>>
        (d_theta, d_gradient, lr, k);
    }

    cudaDeviceSynchronize();
    
    cudaMemcpy(theta_host, d_theta, theta_size, cudaMemcpyDeviceToHost); // Return theta to host
    
    // Destroy streams and handle, free memory allocations.
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cublasDestroy(cublas_handle);
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_theta);
    cudaFree(d_gradient);
    cudaFree(d_y_pred);
    cudaFree(d_error);
    cudaFree(d_loss);
}