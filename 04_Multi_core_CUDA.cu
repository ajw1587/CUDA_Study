#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define NUM_DATA 512

__global__ void vecAdd(int *_a, int *_b, int *_c)
{
	int tID = blockDim.x * blockIdx.x + threadIdx.x;
	_c[tID] = _a[tID] + _b[tID];
}

int main(void)
{
	dim3 dimGrid(NUM_DATA / 256, 1, 1);
	dim3 dimBlock(256, 1, 1);

	vecAdd << <dimGrid, dimBlock >> > (d_a, d_b, d_c);
}