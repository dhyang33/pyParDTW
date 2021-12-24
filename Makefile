all: 
	nvcc  -Xcompiler -fPIC -shared src/ParDTWWrapper.cu src/ParDTW.cu src/Backtrace.cu -o parDTW.so
clean:
	rm -rf *.out
