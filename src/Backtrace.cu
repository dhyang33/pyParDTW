#ifndef __Backtrace__
#define __Backtrace__

#include "ParDTW.hpp"
#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess){
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__device__ unsigned int readBits(unsigned int* back_m, long long int v2_size, long long int x, long long int y){
	long long int bitPos = 2*(y*v2_size+x);
	long long int arrIndex = bitPos/32;
	long long int arrOffset = bitPos % 32;
	unsigned int elem = back_m[arrIndex];
	return (elem >> arrOffset) & 3; // 3==b'11'=2 bits
}

__global__ void BacktraceKernel(unsigned int* back_m, float* frame_m, int v1_size, int v2_size, int* path, bool SUBSEQUENCE){
	// Get starting point
	int cur_row = v1_size - 1;
	int cur_col = -1;
	if (SUBSEQUENCE == 0){
		cur_col = v2_size - 1;
	}
	else{
		float min_val = MAXDOUBLE;
		for(int i = 0; i < v2_size; i++){
			float val = frame_m[i];
			if (val < min_val){
				min_val = val;
				cur_col = i;
			}	
		}
	}
	
	bool done = false;
	while(!done){
		path[cur_row] = cur_col;	
		if (cur_row < 0 or cur_col < 0) break;		
		unsigned int bit = readBits(back_m, v2_size, cur_col, cur_row);
		if (bit == 1){
			cur_row -= 1;
			cur_col -= 1;
		}
		else if (bit == 2){
			cur_row -= 1;
			cur_col -= 2;
		}
		else if(bit == 3){
			cur_row -= 2;
			cur_col -= 1;
		}
		done = (bit == 0);
	}
}

int* Backtrace(DTWStruct accum, size_t v1_size, size_t v2_size, bool SUBSEQUENCE){
	// Allocate final path
	int* cu_final_path;
	size_t final_path_size = v1_size*sizeof(int);
	gpuErrchk(cudaMalloc((void **) &cu_final_path, final_path_size));
	gpuErrchk(cudaMemset(cu_final_path, -1, final_path_size));
	cudaDeviceSynchronize();

	// Free memory
	BacktraceKernel<<<1, 1>>>(accum.cu_back_m, accum.cu_frame_m, v1_size, v2_size, cu_final_path, SUBSEQUENCE);
	cudaDeviceSynchronize();

	// Copy final path to CPU
	int* path = new int[v1_size];
	gpuErrchk(cudaMemcpy(path, cu_final_path, final_path_size, cudaMemcpyDeviceToHost));

	// Free memory
	cudaFree(accum.cu_frame_m);
	cudaFree(accum.cu_back_m);

	return path;
}
#endif
