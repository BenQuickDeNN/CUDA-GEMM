#include "gpu.h"
#include "matrix.h"

#include <cstdio>
#include <cmath>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define F 2.2E3
#define Time 1E6

void test1();
void test2();

inline unsigned long long rdtsc(void)
{
	unsigned long hi = 0, lo = 0;

	__asm__ __volatile__ ("lfence;rdtsc" : "=a"(lo), "=d"(hi));

	return (((unsigned long long)lo))|(((unsigned long long)hi)<<32);
}

int main(int argc, char** argv)
{
    test2();
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
    float alpha = 1.0;
    float beta = 0.0;
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
                result(i, j) += alpha * A(i, k) * B(k, j) + beta * C(i, j);
        }
    double start,end,elapsed;
    std::printf("start cuda computation...\r\n");
    start = rdtsc();
    cuda_matrix_mul_sp(alpha, beta, C, A, B);
    end = rdtsc();
	elapsed= (end - start)/(F * Time);
    std::printf("finish cuda computation, elapse %f s\r\n", elapsed);
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