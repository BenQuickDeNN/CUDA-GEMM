#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * @brief display gpu information
 * @param devProp device
*/
void dispGPUInfo(const cudaDeviceProp& devProp);
/**
 * @brief display gpu information
 * @param dev_id gpu id
 * @return GPU information
*/
cudaDeviceProp getGPUInfo(const unsigned int& dev_id);

cudaDeviceProp getGPUInfo(const unsigned int& dev_id)
{
	std::printf("----------------GPU----------------\r\n");
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, dev_id);
	return devProp;
}
void dispGPUInfo(const cudaDeviceProp& devProp)
{
	std::printf("ʹGPU name: %s\r\n", devProp.name);
	std::printf("number of SMs: %d\r\n", devProp.multiProcessorCount);
	std::printf("max grid size: %d x %d x %d\r\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
	std::printf("size of shared memory per block: %f KB\r\n", devProp.sharedMemPerBlock / 1024.0);
	std::printf("max number of thread per block: %d\r\n", devProp.maxThreadsPerBlock);
	std::printf("max number of thread per SM: %d\r\n", devProp.maxThreadsPerMultiProcessor);
	std::printf("number of block per SM: %d\r\n", devProp.maxThreadsPerMultiProcessor / devProp.maxThreadsPerBlock);
	std::printf("warp size: %d\r\n", devProp.warpSize);
	std::printf("max number of thread per SM per warp size: %d\r\n", 
		devProp.maxThreadsPerMultiProcessor / devProp.warpSize);
}