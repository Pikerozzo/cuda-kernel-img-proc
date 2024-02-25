#include <iostream>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Convolution.cuh"

using namespace std;

const int BLOCK_SIZE = 32;

const int KERNEL_SIZE = 3;
__constant__ float KERNEL_MASK[KERNEL_SIZE * KERNEL_SIZE];

const int MAX_CHANNELS = 4;
const int CHANNELS_MAX_VALUE = 255;

const int TILE_SIZE = BLOCK_SIZE - (KERNEL_SIZE - 1);

Convolution::Convolution() {}

Convolution::~Convolution() {}

__global__ void constantKernel(unsigned char* in, unsigned char* out, int width, int height, int channels, int pixelNormValue)
{
	// get pixel coordinates for the current thread 
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// check if the thread is within the valid image range
	if (row < height && col < width)
	{
		// define the starting position of the convolution mask
		int maskStartRow = row - (KERNEL_SIZE / 2);
		int maskStartCol = col - (KERNEL_SIZE / 2);

		// iterate over color channels
		for (int c = 0; c < channels; c++)
		{
			float pixelVal = 0;

			// iterate over the elements of the convolution mask
			for (int y = 0; y < KERNEL_SIZE; ++y) {
				for (int x = 0; x < KERNEL_SIZE; ++x)
				{
					// calculate the current position in the input image
					int curRow = min(height - 1, max(maskStartRow + y, 0));
					int curCol = min(width - 1, max(maskStartCol + x, 0));

					// perform convolution by multiplying the pixel value with the corresponding kernel value
					pixelVal += in[(curRow * width + curCol) * channels + c] * KERNEL_MASK[y * KERNEL_SIZE + x];
				}
			}

			// normalize the pixel value, then store it in the output image
			pixelVal = min((float)CHANNELS_MAX_VALUE, max(pixelVal + pixelNormValue, 0.0f));
			out[(row * width + col) * channels + c] = (unsigned char)pixelVal;
		}
	}
}

__global__ void globalKernel(unsigned char* in, unsigned char* out, int width, int height, int channels, float* mask, int pixelNormValue)
{
	// get pixel coordinates for the current thread 
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// check if the thread is within the valid image range
	if (row < height && col < width)
	{
		// define the starting position of the convolution mask
		int maskStartRow = row - (KERNEL_SIZE / 2);
		int maskStartCol = col - (KERNEL_SIZE / 2);

		// iterate over color channels
		for (int c = 0; c < channels; c++)
		{
			float pixelVal = 0;

			// iterate over the elements of the convolution mask
			for (int y = 0; y < KERNEL_SIZE; ++y) {
				for (int x = 0; x < KERNEL_SIZE; ++x)
				{
					// calculate the current position in the input image
					int curRow = min(height - 1, max(maskStartRow + y, 0));
					int curCol = min(width - 1, max(maskStartCol + x, 0));

					// perform convolution by multiplying the pixel value with the corresponding kernel value
					pixelVal += in[(curRow * width + curCol) * channels + c] * mask[y * KERNEL_SIZE + x];
				}
			}

			// normalize the pixel value, then store it in the output image
			pixelVal = min((float)CHANNELS_MAX_VALUE, max(pixelVal + pixelNormValue, 0.0f));
			out[(row * width + col) * channels + c] = (unsigned char)pixelVal;
		}
	}
}

__global__ void sharedKernel(unsigned char* in, unsigned char* out, int width, int height, int channels, int pixelNormValue)
{
	// variable used to create a tile for storing a portion of the image in the shared memory
	extern __shared__ unsigned char sharedData[];

	// get thread indices
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// get the output indices
	int row_o = ty + blockIdx.y * TILE_SIZE;
	int col_o = tx + blockIdx.x * TILE_SIZE;

	// shift to obtain input indices considering the convolution mask size
	int row_i = row_o - (KERNEL_SIZE / 2);
	int col_i = col_o - (KERNEL_SIZE / 2);

	// load tile elements from the input image into shared memory
	for (int c = 0; c < channels; c++)
	{
		unsigned char pixelVal = 0;

		// ensure that the input indices are within the valid image range
		row_i = min(height - 1, max(row_i, 0));
		col_i = min(width - 1, max(col_i, 0));

		// store the loaded pixel value in the shared memory tile
		sharedData[(ty * BLOCK_SIZE + tx) * channels + c] = in[(row_i * width + col_i) * channels + c];
	}

	// wait for all tile elements to be loaded
	__syncthreads();

	// perform convolution using the shared memory tile and convolution mask
	// only compute if the current thread is part of an output tile element
	if (tx < TILE_SIZE && ty < TILE_SIZE && row_o < height && col_o < width) {
		for (int c = 0; c < channels; c++)
		{
			float pixelVal = 0;

			// iterate over the elements of the convolution mask
			for (int y = 0; y < KERNEL_SIZE; y++)
				for (int x = 0; x < KERNEL_SIZE; x++)
					pixelVal += sharedData[((y + ty) * BLOCK_SIZE + x + tx) * channels + c] * KERNEL_MASK[y * KERNEL_SIZE + x];

			// normalize the pixel value, then store it in the output image
			pixelVal = min((float)CHANNELS_MAX_VALUE, max(pixelVal + pixelNormValue, 0.0f));
			out[(row_o * width + col_o) * channels + c] = pixelVal;
		}
	}
}

void Convolution::applyConstant(Image& image, Kernel& kernel) {

	// get the total size of the image and kernel mask
	size_t imageSize = image.getTotalSize(true);
	size_t kernelSize = kernel.getTotalSize(true);

	// copy the kernel mask to the constant memory
	cudaError_t cudaError = cudaMemcpyToSymbol(KERNEL_MASK, kernel.getKernelFilter(), kernelSize);
	// check for CUDA errors
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(cudaError));
		return;
	}

	// allocate device memory for input and output images and copy input image data to the device
	unsigned char* dev_imgIn;
	unsigned char* dev_imgOut;
	unsigned char* imgOut = (unsigned char*)malloc(imageSize);
	cudaMalloc((void**)&dev_imgIn, imageSize);
	cudaMalloc((void**)&dev_imgOut, imageSize);
	cudaMemcpy(dev_imgIn, image.getImageData(), imageSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_imgOut, imgOut, imageSize, cudaMemcpyHostToDevice);

	// define the grid and block dimensions
	dim3 gridSize(ceil((float)image.getWidth() / (float)BLOCK_SIZE), ceil((float)image.getHeight() / (float)BLOCK_SIZE));
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

	// launch the CUDA kernel for constant memory convolution
	constantKernel << <gridSize, blockSize >> > (dev_imgIn, dev_imgOut, image.getWidth(), image.getHeight(), image.getChannels(), kernel.getFilterNormalizationValue());

	// check for CUDA errors
	cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		printf("CUDA error: %s", cudaGetErrorString(cudaError));
	}

	// synchronize device to ensure completion of the kernel
	cudaDeviceSynchronize();

	// copy the image result back to the host and update the image data
	cudaMemcpy(imgOut, dev_imgOut, imageSize, cudaMemcpyDeviceToHost);
	image.setImageData(imgOut);

	// free allocated device memory
	cudaFree(dev_imgIn);
	cudaFree(dev_imgOut);
}

void Convolution::applyGlobal(Image& image, Kernel& kernel) {

	// get the total size of the image and kernel mask
	size_t imageSize = image.getTotalSize(true);
	size_t kernelSize = kernel.getTotalSize(true);

	// allocate device memory and copy input and output images and kernel mask 
	unsigned char* dev_imgIn;
	unsigned char* dev_imgOut;
	float* dev_kernelFilter;
	unsigned char* imgOut = (unsigned char*)malloc(imageSize);
	cudaMalloc((void**)&dev_imgIn, imageSize);
	cudaMalloc((void**)&dev_imgOut, imageSize);
	cudaMalloc((void**)&dev_kernelFilter, kernelSize);
	cudaMemcpy(dev_imgIn, image.getImageData(), imageSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_imgOut, imgOut, imageSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_kernelFilter, kernel.getKernelFilter(), kernelSize, cudaMemcpyHostToDevice);

	// define the grid and block dimensions
	dim3 gridSize(ceil((float)image.getWidth() / (float)BLOCK_SIZE), ceil((float)image.getHeight() / (float)BLOCK_SIZE));
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

	// launch the CUDA kernel for global memory convolution
	globalKernel << <gridSize, blockSize >> > (dev_imgIn, dev_imgOut, image.getWidth(), image.getHeight(), image.getChannels(), dev_kernelFilter, kernel.getFilterNormalizationValue());
	
	// check for CUDA errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: %s", cudaGetErrorString(err));
	}

	// synchronize device to ensure completion of the kernel
	cudaDeviceSynchronize();

	// copy the image result back to the host and update the image data
	cudaMemcpy(imgOut, dev_imgOut, imageSize, cudaMemcpyDeviceToHost);
	image.setImageData(imgOut);

	// free allocated device memory
	cudaFree(dev_imgIn);
	cudaFree(dev_imgOut);
	cudaFree(dev_kernelFilter);
}

void Convolution::applyShared(Image& image, Kernel& kernel) {

	// get the total size of the image and kernel mask
	size_t imageSize = image.getTotalSize(true);
	size_t kernelSize = kernel.getTotalSize(true);

	// copy the kernel mask to the constant memory
	cudaError_t cudaError = cudaMemcpyToSymbol(KERNEL_MASK, kernel.getKernelFilter(), kernelSize);
	// check for CUDA errors
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(cudaError));
		return;
	}

	// allocate device memory and copy input and output images
	unsigned char* dev_imgIn;
	unsigned char* dev_imgOut;
	unsigned char* imgOut = (unsigned char*)malloc(imageSize);
	cudaMalloc((void**)&dev_imgIn, imageSize);
	cudaMalloc((void**)&dev_imgOut, imageSize);
	cudaMemcpy(dev_imgIn, image.getImageData(), imageSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_imgOut, imgOut, imageSize, cudaMemcpyHostToDevice);

	// define the grid and block dimensions
	dim3 gridSize(ceil(image.getWidth() / (float)TILE_SIZE), ceil(image.getHeight() / (float)TILE_SIZE), 1);
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

	// calculate shared memory size
	size_t sharedMemorySize = BLOCK_SIZE * BLOCK_SIZE * image.getChannels() * sizeof(unsigned char);

	// launch the CUDA kernel for shared memory convolution
	sharedKernel << <gridSize, blockSize, sharedMemorySize >> > (dev_imgIn, dev_imgOut, image.getWidth(), image.getHeight(), image.getChannels(), kernel.getFilterNormalizationValue());

	// check for CUDA errors
	cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		printf("CUDA error: %s", cudaGetErrorString(cudaError));
	}

	// synchronize device to ensure completion of the kernel
	cudaDeviceSynchronize();

	// copy the image result back to the host and update the image data
	cudaMemcpy(imgOut, dev_imgOut, imageSize, cudaMemcpyDeviceToHost);
	image.setImageData(imgOut);

	// free allocated device memory
	cudaFree(dev_imgIn);
	cudaFree(dev_imgOut);
}

void Convolution::applySequential(Image& image, Kernel& kernel)
{
	// allocate output image
	unsigned char* out = (unsigned char*)malloc(image.getTotalSize(true));

	// iterate over all image pixels
	for (int row = 0; row < image.getHeight(); row++)
	{
		for (int col = 0; col < image.getWidth(); col++)
		{
			// define the starting position of the convolution mask
			int maskStartRow = row - (KERNEL_SIZE / 2);
			int maskStartCol = col - (KERNEL_SIZE / 2);

			// iterate over color channels
			for (int c = 0; c < image.getChannels(); c++)
			{
				float pixelVal = 0;

				// iterate over the elements of the convolution mask
				for (int i = 0; i < KERNEL_SIZE; ++i) {
					for (int j = 0; j < KERNEL_SIZE; ++j)
					{
						// calculate the current position in the input image
						int curRow = min(image.getHeight() - 1, max(maskStartRow + i, 0));
						int curCol = min(image.getWidth() - 1, max(maskStartCol + j, 0));

						// check if the calculated position is within the valid image range
						if (curRow >= 0 && curRow < image.getHeight() && curCol >= 0 && curCol < image.getWidth())
						{
							// perform convolution by multiplying the pixel value with the corresponding kernel value
							pixelVal += image.getImageData()[(curRow * image.getWidth() + curCol) * image.getChannels() + c] * kernel.getKernelFilter()[i * KERNEL_SIZE + j];
						}
					}
				}

				// normalize the pixel value, then store it in the output image
				pixelVal = min((float)CHANNELS_MAX_VALUE, max(pixelVal + kernel.getFilterNormalizationValue(), 0.0f));
				out[(row * image.getWidth() + col) * image.getChannels() + c] = (unsigned char)pixelVal;
			}
		}
	}

	// update the image data
	image.setImageData(out);
}

void Convolution::apply(Image& image, Kernel& kernel, ExecutionMode execMode)
{
	try
	{
		switch (execMode)
		{
		case ExecutionMode::SEQUENTIAL:
		{
			applySequential(image, kernel);
			break;
		}
		case ExecutionMode::CONSTANT:
		{
			applyConstant(image, kernel);
			break;
		}
		case ExecutionMode::GLOBAL:
		{
			applyGlobal(image, kernel);
			break;
		}
		default:
		case ExecutionMode::SHARED:
		{
			applyShared(image, kernel);
			break;
		}
		}
	}
	catch (const std::exception& e)
	{
		printf("Error: %s\n", e.what());
	}
}

void Convolution::resetCuda(ExecutionMode execMode)
{
	// reset the GPU device
	if (execMode != ExecutionMode::SEQUENTIAL)
	{
		cudaDeviceReset();
	}
}