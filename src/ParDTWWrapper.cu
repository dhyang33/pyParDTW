#include <vector>
#include <utility>
#include <iostream>
#include "ParDTW.hpp"

extern "C" {
	int* _parDTW(double* v1, double* v2, size_t v1_size, size_t v2_size, size_t hidden_size, bool SUBSEQUENCE);
	void freeint (int* v);
	void freedouble (double* v);
}
int* _parDTW(double* v1, double* v2, size_t v1_size, size_t v2_size, size_t hidden_size, bool SUBSEQUENCE) {
	DTWStruct accum = ParDTW(v1, v2, v1_size, v2_size, hidden_size, SUBSEQUENCE);
	int* path = Backtrace(accum, v1_size, v2_size, SUBSEQUENCE);	
	return path;
}
void freeint(int* v){
	delete v;
}
void freedouble(double* v){
	delete v;
}
