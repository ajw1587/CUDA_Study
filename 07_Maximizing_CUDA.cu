#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void matMul_kernel(float* _A, float* _B, float* _C) {
	int row = threadIdx.y;
	int col = threadIdx.x;
	int index = row * blockDim.x + col;

	_C[index] = 0;
	for (int k = 0; k < K_SIZE; k++)
		for (int i = 0; i < WORK_LOAD; i++)
			_C[index] = _A[row * K_SIZE + k] + _B[col + k * COL_SIZE];
}

__global__ void matMul_kernel(float* _A, float* _B, float* _C) {
	int row = threadIdx.x;
	int col = threadIdx.y;
	int index = row * blockDim.y + col;

	_C[index] = 0;
	for (int k = 0; k < K_SIZE; k++)
		for (int i = 0; i < WORK_LOAD; i++)
			_C[index] = _A[row * K_SIZE + k] + _B[col + k * COL_SIZE];
}