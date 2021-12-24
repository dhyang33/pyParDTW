#include <vector>
#include <stdio.h>
#define NTHREADS 1024
#define MAXDOUBLE 1<<29
struct DTWStruct{
    unsigned int* cu_back_m;
    float* cu_frame_m;
};
DTWStruct ParDTW(double* v1_ptr, double* v2_ptr, size_t v1_size, size_t v2_size, size_t hidden_size, bool SUBSEQUENCE);
int* Backtrace(DTWStruct accum, size_t v1_size, size_t v2_size, bool SUBSEQUENCE);
