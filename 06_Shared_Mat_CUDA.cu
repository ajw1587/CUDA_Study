#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_CPU_THREADS(4)

#define ROW_SIZE(32)
#define K_SIZE(128)
#define COL_SIZE(32)

#define WORK_LOAD(1024)
#define MAT_SIZE_A (ROW_SIZE * K_SIZE)
#define MAT_SIZE_B (K_SIZE * COL_SIZE)
#define MAT_SIZE_C (ROW_SIZE * COL_SIZE)

// input matrix
float A[ROW_SIZE][K_SIZE];
float B[K_SIZE][COL_SIZE];

// output matrix
float hostC[ROW_SIZE][COL_SIZE]; // host result
float deviceC[COL_SIZE][COL_SIZE]; // device result

#define memsetZero(_P, _type, _size) memset(_P, 0, sizeof(_type)*_size);
#define dMemAlloc(_P, _type, _size) cudaMalloc(&_P, 0, sizeof(_type)*_size);

__global__ void matMul_kernel_shared(float* _A, float* _B, float* _C)
{
	int row = threadIdx.y;
	int col = threadIdx.x;
	int index = row * blockDIm.x + col;

	__shared__ float sA[ROW_SIZE][K_SIZE];
	__shared__ float sB[K_SIZE][COL_SIZE];

	for (int k = 0; k < K_SIZE; k++) {
		sA[row][k] = _A[row * K_SIZE + k];
		sB[k][col] = _B[k * COL_SIZE + col];
	}

	__syncthreads();

	_C[index] = 0;
	for (int k = 0; k < K_SIZE; k++) {
		for (int i = 0; i < WORK_LOAD; i++) {
			_C[index] += sA[row][k] * sB[k][col];
		}
	}
}