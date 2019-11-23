#include "gpu.h"
#include "matrix.h"

#include <cstdio>
#include <cmath>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void test1();
void test2();
int main(int argc, char** argv)
{
    test1();
    return 0;
}
void test1()
{
    cudaDeviceProp devProp;
    int count;
    cudaGetDeviceCount(&count);
    for (int i = 0; i < count; i++)
    {
        cudaGetDeviceProperties(&devProp, i);
        dispGPUInfo(devProp);
    }
}
void test2()
{
    const unsigned int msize = 1024;
    gemm::MatrixSP C(msize, msize);
    gemm::MatrixSP A(msize, msize);
    gemm::MatrixSP B(msize, msize);
    gemm::MatrixSP result(msize, msize);
    /* initialize value */
    for (int i = 0; i < msize; i++)
        for (int j = 0; j < msize; j++)
        {
            A(i, j) = 10.0 * ((float) (rand() % 10) / 9);
            B(i, j) = 10.0 * ((float) (rand() % 10) / 9);
        }
    for (int i = 0; i < msize; i++)
        for (int j = 0; j < msize; j++)
        {
            result(i, j) = 0.0;
            for (int k = 0; k < B.width(); k++)
                result(i, j) += A(i, k) * B(k, j);
        }
    std::printf("start cuda computation...\r\n");
    cuda_matrix_mul_sp(C, A, B);
    std::printf("finish cuda computation...\r\n");
    //gemm::MatrixSP::mul(C, A, B);
    /* checking result */
    for (int i = 0; i < msize; i++)
        for (int j = 0; j < msize; j++)
            if (std::abs(C(i, j) - result(i, j)) > 0.2)
            {
                std::fprintf(stderr, "computation error!\r\n");
                return;
            }
    std::printf("computation corrected\r\n");
}