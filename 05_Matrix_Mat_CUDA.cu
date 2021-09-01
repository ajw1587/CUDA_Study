#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void vecAdd()
{
	int row = threadIdx.y;
	int col = threadIdx.x;
	int index = row * blockDim.x + col;

	_C[index] = 0;
	for (int k = 0; k < K_SIZE; k++)
		_C[index] += _A[row * K_SIZE + k] * _B[col + k * COL_SIZE];
}

int main(void)
{
	
}