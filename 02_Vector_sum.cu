#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_DATA 512

__global__ void vecAdd(int* _a, int* _b, int* _c) {
	int tID = threadIdx.x;
	_c[tID] = _a[tID] + _b[tID];
}

int main(void)
{
	// 변수 선언!
	int* a, *b, *c;
	int* d_a, *d_b, *d_c;
	int memSize = sizeof(int)*NUM_DATA;
	printf("%d elements, memSize = %d bytes\n", NUM_DATA, memSize);
	
	// a, b, c 초기화!
	a = new int[NUM_DATA]; memset(a, 0, memSize);
	b = new int[NUM_DATA]; memset(b, 0, memSize);
	c = new int[NUM_DATA]; memset(c, 0, memSize);

	// a, b 랜덤값으로 채워주기!
	for (int i = 0; i < NUM_DATA; i++) {
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}

	// GPU에 공간 할당!
	cudaMalloc(&d_a, memSize);
	cudaMalloc(&d_b, memSize);
	cudaMalloc(&d_c, memSize);

	// GPU 공간에 데이터 복사!
	cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);

	vecAdd << <1, NUM_DATA >> > (d_a, d_b, d_c);

	cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);

	// 결과 확인하기
	bool result = true;
	for (int i = 0; i < NUM_DATA; i++) {
		if ((a[i] + b[i]) != c[i]) {
			printf("[%d] The results is not matched! (%d, %d)\n",
					i, a[i]+b[i], c[i]);

			result = false;
		}
	}

	if (result)
		printf("GPU works well!\n");

	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	delete[] a; delete[] b; delete[] c;

	return 0;
}