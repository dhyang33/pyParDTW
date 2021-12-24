# SegmentalDTWCUDA
This respository contains code for cuda-accelerated segmental DTW.

To change parameters M,N edit src/SegmentDTW.hpp

To recompile, run make. To run the code, 
```
./segDTW.out -v1=file1.txt -v2=file2.txt
```

Please take a look at generator.py to figure out the format of the input files.

For python code integration, consider using the subprocess library.
```
p = Popen(["./segDTW.out -v1=file1.txt -v2=file2.txt"], shell=True)
```
