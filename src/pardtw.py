#!/usr/bin/env python3
import numpy as np
import ctypes
from ctypes import POINTER, c_int, c_double, c_size_t, c_bool
from numpy.ctypeslib import ndpointer

def pardtw(v1, v2, subsequence=False):
    pardtwlib = ctypes.CDLL("./parDTW.so")
    
    v1_size = v1.shape[0]
    v2_size = v2.shape[0]
    try:
        assert(v1.shape[1]==v2.shape[1])
    except:
        print("hidden dim mismatch")
        return
    hidden_size = v1.shape[1]
    
    v1_ptr = np.array(v1).flatten()
    v2_ptr = np.array(v2).flatten()

    v1_ptr = v1_ptr.ctypes.data_as(POINTER(c_double))
    v2_ptr = v2_ptr.ctypes.data_as(POINTER(c_double))
    
    pardtwlib._parDTW.argtypes = [POINTER(c_double), POINTER(c_double), c_size_t, c_size_t, c_size_t, c_bool]
    pardtwlib._parDTW.restype = ndpointer(dtype=c_int, shape=(v1_size,))
    path = pardtwlib._parDTW(v1_ptr, v2_ptr, v1_size, v2_size, hidden_size, subsequence)
    final_path = []
    for x, y in enumerate(path):
        if y != -1:
            final_path.append([x, y])
    return np.array(final_path)
