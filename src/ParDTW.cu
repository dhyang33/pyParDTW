#ifndef __segDTW__
#define __segDTW__

#include <math.h>
#include <assert.h>
#include <iostream>
#include "ParDTW.hpp"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess){
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__device__ void setBits(unsigned int* back_m, long long int v2_size, long long int x, long long int y, unsigned int bits){
	long long int bitPos = 2*(y*v2_size+x);
	long long int arrIndex = bitPos/32;
	long long int arrOffset = bitPos % 32;
	unsigned int shiftedBits = bits << arrOffset;
	atomicOr(back_m + arrIndex, shiftedBits);
}

__global__ void DTWKernel(double* v1_ptr, double* v2_ptr, int diagonalIdx, int hidden_size, double* fringe, int max_diagonal_size, unsigned int* back_m, float* frame_m, size_t v1_size, size_t v2_size, int fparity, bool SUBSEQUENCE){
	// Figure out what cell we are at
	int responsibility = (int) blockIdx.x;
	int e = threadIdx.x+NTHREADS*responsibility;
	
	// Figure out what element we are at
	int global_x = diagonalIdx - e;
	int global_y = e;

	// Check Bounds
	if (global_x < 0 or global_x >= v2_size) return;
	if (global_y < 0 or global_y >= v1_size) return;

	// Calculate cost
	double v1_sum = 0;
	double v2_sum = 0;
	double dot = 0;
	int v1_idx = global_y;
	int v2_idx = global_x;
	for(int i = 0; i < hidden_size; i++){
		v1_sum+=v1_ptr[hidden_size*v1_idx+i]*v1_ptr[hidden_size*v1_idx+i];
		v2_sum+=v2_ptr[hidden_size*v2_idx+i]*v2_ptr[hidden_size*v2_idx+i];
		dot+=v1_ptr[hidden_size*v1_idx+i]*v2_ptr[hidden_size*v2_idx+i];
	}
	double cost = 1-dot/(sqrt(v1_sum)*sqrt(v2_sum));

	// Optimization
	double option1 = MAXDOUBLE;
	double option2 = MAXDOUBLE;
	double option3 = MAXDOUBLE;
	// Bottom row of DTW
	if (SUBSEQUENCE == 0 and global_y == 0) option1 = cost; // bottom row of subsequence DTW
	else if (SUBSEQUENCE == 1 and global_x == 0 and global_y == 0) option1 = cost;
	else {
		int o1_idx = e - 1; // (-1,-1)
		int o2_idx = e - 1; // (-2, -1)
		int o3_idx = e - 2; // (-1, -2)
		if (SUBSEQUENCE) {
			if (global_x-1 >= 0 && global_y-1 >= 0) option1 = cost*1 + fringe[max_diagonal_size*((fparity+2)%4)+o1_idx];
			if (global_x-2 >= 0 && global_y-1 >= 0) option2 = cost*1 + fringe[max_diagonal_size*((fparity+3)%4)+o2_idx];
			if (global_x-1 >= 0 && global_y-2 >= 0) option3 = cost*2 + fringe[max_diagonal_size*((fparity+3)%4)+o3_idx];
		}
		else{
			if (global_x-1 >= 0 && global_y-1 >= 0) option1 = cost*2 + fringe[max_diagonal_size*((fparity+2)%4)+o1_idx];
			if (global_x-2 >= 0 && global_y-1 >= 0) option2 = cost*3 + fringe[max_diagonal_size*((fparity+3)%4)+o2_idx];
			if (global_x-1 >= 0 && global_y-2 >= 0) option3 = cost*3 + fringe[max_diagonal_size*((fparity+3)%4)+o3_idx];
		}
	}
	if (option1 > MAXDOUBLE) option1 = MAXDOUBLE;
	if (option2 > MAXDOUBLE) option2 = MAXDOUBLE;
	if (option3 > MAXDOUBLE) option3 = MAXDOUBLE;
	double best_option = MAXDOUBLE;
	unsigned int back_idx=0;
	if (option1 <= option2 && option1 <= option3){
		best_option = option1;
		back_idx = 1;
	}
	else if (option2 <= option1 && option2 <= option3){
		best_option = option2;
		back_idx = 2;
	}
	else{
		best_option = option3;
		back_idx = 3;
	}
	setBits(back_m, v2_size, global_x, global_y, back_idx);
	if (global_y == v1_size-1) frame_m[global_x] = best_option; // log into frame cost matrix
	fringe[max_diagonal_size*((fparity)%4)+e] = best_option;
}

DTWStruct ParDTW(double* v1_ptr, double* v2_ptr, size_t v1_size, size_t v2_size, size_t hidden_size, bool SUBSEQUENCE){
	// Copy vectors to GPU
	double *cu_v1_ptr;
	double *cu_v2_ptr;
	size_t v1_GPU_size = v1_size * hidden_size * sizeof(double);
	size_t v2_GPU_size = v2_size * hidden_size * sizeof(double);
	cudaMalloc((void **) &cu_v1_ptr, v1_GPU_size);
	cudaMalloc((void **) &cu_v2_ptr, v2_GPU_size);
	cudaMemcpy(cu_v1_ptr, v1_ptr, v1_GPU_size, cudaMemcpyHostToDevice);
	cudaMemcpy(cu_v2_ptr, v2_ptr, v2_GPU_size, cudaMemcpyHostToDevice);

	// Calculate cell dimensions
	size_t cell_width = v2_size;
	size_t cell_height = v1_size;

	// Calculate Fringe and copy to GPU
	int max_diagonal_size = min(cell_height, cell_width);	
	int num_responsibilities = ceil((double) max_diagonal_size/NTHREADS);
	assert(NTHREADS*num_responsibilities >= max_diagonal_size);

	int fparity = 0;
	size_t fringe_GPU_size = max_diagonal_size*sizeof(double)*4;
	double* cu_fringe;
	cudaMalloc((void **) &cu_fringe, fringe_GPU_size);

	// Copy Backtrace Matrix to GPU
	unsigned int* cu_back_m; //32 bits (4 bytes)
	size_t back_GPU_size = ceil(v1_size*v2_size/16)*sizeof(unsigned int); //2 bits per element
	cudaMalloc((void **) &cu_back_m, back_GPU_size);
	cudaMemset(cu_back_m, 0, back_GPU_size);

	// Allocate frame matrices
	float* cu_frame_m;
	size_t frame_m_size = v2_size*sizeof(float);
	cudaMalloc((void **) &cu_frame_m, frame_m_size);
	//cudaMemset(cu_frame_m, 0, frame_m_size);
	printf("Finished Allocation\n");
	// Call DTW Kernel
	cudaDeviceSynchronize();
	int num_diagonals = cell_height + cell_width - 1;
	for(int d = 0; d < num_diagonals; d++){
		DTWKernel<<<num_responsibilities, NTHREADS>>>(cu_v1_ptr, cu_v2_ptr, d, hidden_size, cu_fringe, max_diagonal_size, cu_back_m, cu_frame_m, v1_size, v2_size, fparity, SUBSEQUENCE);
		fparity = (fparity + 3)%4;
		cudaDeviceSynchronize();
	}

	// Free cuda memory
	cudaFree(cu_v1_ptr);
	cudaFree(cu_v2_ptr);
	cudaFree(cu_fringe);

	DTWStruct accum;
	accum.cu_back_m = cu_back_m;
	accum.cu_frame_m = cu_frame_m;	

	return accum;
}
#endif
