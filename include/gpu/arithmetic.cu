#include "matrix.h"
#include "gpuinfo.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * @brief cuda kernal -- matrix mul C = alpha * A * B + beta * C, single precision
 * @param alpha coefficient parameter
 * @param beta coefficient parameter
 * @param C result
 * @param A matrix A
 * @param B matrix B
*/
__global__ void kernel_matrix_mul_sp(float alpha, float beta,
    float* C, float* A, float* B, 
    unsigned int widthC, unsigned int widthA, unsigned int widthB)
{
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    const int indexC = row * widthC + col;
    const int indexA = row * widthA;
    C[indexC] = 0.0;
    for (int i = 0; i < widthA; i++)
		C[indexC] += alpha * A[indexA + i] * B[i * widthB + col] + beta * C[indexC];
}

/**
 * @brief cuda -- matrix mul, single precision
 * @param alpha coefficient parameter
 * @param beta coefficient parameter
 * @param C result
 * @param A matrix A
 * @param B matrix B
*/
void cuda_matrix_mul_sp(const float& alpha, const float& beta,
    gemm::MatrixSP& C, gemm::MatrixSP& A, gemm::MatrixSP& B);

void cuda_matrix_mul_sp(const float& alpha, const float& beta,
    gemm::MatrixSP& C, gemm::MatrixSP& A, gemm::MatrixSP& B)
{
    if (A.empty() || B.empty() || C.empty())
    {
        std::fprintf(stderr, "cuda matrix mul sp error, matrix is empty!\r\n");
        return;
    }
    if (A.width() != B.height() || A.height() != C.height() || B.width() != C.width())
    {
        std::fprintf(stderr, "cuda matrix mul sp error, matrix doesnot match!\r\n");
        return;
    }
    /* initialize gpu preference */
    cudaDeviceProp devProp;
    int gpu_id = std::atoi(std::getenv("CUDA_VISIBLE_DEVICES"));
    int gpucount;
    cudaGetDeviceCount(&gpucount);
    if (gpu_id < 0 || gpu_id >= gpucount)
    {
        std::fprintf(stderr, "cuda matrix mul sp error, gpu %d doesnot exist!\r\n", gpu_id);
        return;
    }
    cudaGetDeviceProperties(&devProp, gpu_id);
    const int blockSize = devProp.maxThreadsPerBlock;
    const int blockLen = (int)std::floor(std::sqrt((double)blockSize));
    dim3 cudaBlockSize(blockLen, blockLen);
    dim3 cudaGridSize((C.height() + cudaBlockSize.x - 1) / cudaBlockSize.x, 
        (C.width() + cudaBlockSize.y - 1) / cudaBlockSize.y);
    /* allocate memory on gpu */
    float *cuC, *cuA, *cuB;
    cudaMalloc((void**)&cuC, C.height() * C.width() * sizeof(float));
    cudaMalloc((void**)&cuA, A.height() * A.width() * sizeof(float));
    cudaMalloc((void**)&cuB, B.height() * B.width() * sizeof(float));
    /* copy data */
    cudaMemcpy(cuA, (A._element), A.height() * A.width() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuB, (B._element), B.height() * B.width() * sizeof(float), cudaMemcpyHostToDevice);
    /* execute kernel */
    kernel_matrix_mul_sp<<<cudaGridSize, cudaBlockSize>>>(alpha, beta, cuC, cuA, cuB, C.width(), A.width(), B.width());
    /* copy data */
    cudaMemcpy(C._element, cuC, C.height() * C.width() * sizeof(float), cudaMemcpyDeviceToHost);
    /* free memory on gpu */
    cudaFree(cuC);
    cudaFree(cuA);
    cudaFree(cuB);
}
