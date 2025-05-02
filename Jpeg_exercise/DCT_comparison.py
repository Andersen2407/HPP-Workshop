import numpy as np
import time
from numba import njit, prange

import cupy as cp
from scipy.fft import dct

import matplotlib.pyplot as plt


# basic timer decorator
def timer(f):                                   # Decorator to calculate execution time of a funciton
    def wrapper(*args, **kw):                   # Needed to decorate a function with input arguments
        t_start = time.time()
        result = f(*args, **kw)                 # Calling function
        t_end = time.time()
        return result, t_end-t_start            # Return the result AND the execution time
    return wrapper

# ======================================= Parallel implementations ======================================
@timer
@njit(parallel=True)
def dct_2d_jit_parallel(block: np.ndarray):
    N = block.shape[0]  # Assuming block is 8x8
    
    # Apply 1D DCT on rows
    y_rows = np.zeros_like(block, dtype=np.float32)
    for row in prange(block.shape[0]):
        # Extract the row
        x = block[row, :]
        
        # For every y_k
        for k in prange(y_rows.shape[1]):
            # Type-II DCT formula right here! (for one y_k entry)
            for n in prange(y_rows.shape[1]):        # summation happens here
                y_rows[row, k] += x[n] * np.cos(np.pi*k*(2 * n + 1) / (2*N))
            y_rows[row, k] *= 2
        # -------------------------------
    
    # Apply 1D DCT on columns
    y_columns = np.zeros_like(block, dtype=np.float32)
    for column in prange(block.shape[1]):
        # Apply DCT formula for each column
        x = y_rows[:, column]
        
        for k in prange(N):
            # Type-II DCT formula right here!
            for n in prange(N):
                y_columns[k, column] += x[n] * np.cos(np.pi*k*(2 * n + 1) / (2*N))
            y_columns[k, column] *= 2
        # -------------------------------
    
    matrix_y = y_columns
    return matrix_y

@timer
def dct_cuda_parallel_scipy(block: np.ndarray):
    # Apply 2D DCT (Type-II) by chaining 1D DCTs
    dct_rows = dct(block, type=2, norm='ortho', axis=0)
    dct_cols = dct(dct_rows, type=2, norm='ortho', axis=1)

    return dct_cols

# Uses cuda through library
@timer
def dct_cuda_parallel(block: np.ndarray):
    block = cp.array(block)     # Allocate on GPU

    N = block.shape[0]  # Assuming block is 8x8
    # Apply 1D DCT on rows
    y_rows = cp.zeros_like(block, dtype=np.float32)
    for row in range(block.shape[0]):
        # Extract the row
        x = block[row, :]
        
        # For every y_k
        for k in range(y_rows.shape[1]):
            # Type-II DCT formula right here! (for one y_k entry)
            for n in range(y_rows.shape[1]):        # summation happens here
                y_rows[row, k] += x[n] * cp.cos(cp.pi*k*(2 * n + 1) / (2*N))
            y_rows[row, k] *= 2
        # -------------------------------
    
    # Apply 1D DCT on columns
    y_columns = cp.zeros_like(block, dtype=cp.float32)
    for column in range(block.shape[1]):
        # Apply DCT formula for each column
        x = y_rows[:, column]
        
        for k in range(N):
            # Type-II DCT formula right here!
            for n in range(N):
                y_columns[k, column] += x[n] * cp.cos(cp.pi*k*(2 * n + 1) / (2*N))
            y_columns[k, column] *= 2
        # -------------------------------
    
    matrix_y = y_columns

    cpu_block = cp.asnumpy(matrix_y)   # download to RAM from GPU

    return cpu_block

@timer
def dct_cuda_manual(block: np.ndarray):
    block_gpu = cp.array(block, dtype=cp.float32)
    N = block_gpu.shape[0]

    k = cp.arange(N).reshape(-1, 1)
    n = cp.arange(N).reshape(1, -1)

    # Create normalized DCT-II transform matrix (orthonormal basis)
    dct_matrix = cp.cos(cp.pi * (2*n + 1) * k / (2*N)) * cp.sqrt(2 / N)
    dct_matrix[0, :] /= cp.sqrt(2)

    # Apply DCT on rows
    temp = dct_matrix @ block_gpu

    # Apply DCT on columns
    result = temp @ dct_matrix.T

    return cp.asnumpy(result)
# ====================================== Parallel implementations ======================================




# ====================================== Sequential implementations ======================================
@timer
@njit()
def dct_2d_jit_sequential(block: np.ndarray):
    N = block.shape[0]  # Assuming block is 8x8
    
    # Apply 1D DCT on rows
    y_rows = np.zeros_like(block, dtype=np.float32)
    for row in range(block.shape[0]):
        # Extract the row
        x = block[row, :]
        
        # For every y_k
        for k in range(y_rows.shape[1]):
            # Type-II DCT formula right here! (for one y_k entry)
            for n in range(y_rows.shape[1]):        # summation happens here
                y_rows[row, k] += x[n] * np.cos(np.pi*k*(2 * n + 1) / (2*N))
            y_rows[row, k] *= 2
        # -------------------------------
    
    # Apply 1D DCT on columns
    y_columns = np.zeros_like(block, dtype=np.float32)
    for column in range(block.shape[1]):
        # Apply DCT formula for each column
        x = y_rows[:, column]
        
        for k in range(N):
            # Type-II DCT formula right here!
            for n in range(N):
                y_columns[k, column] += x[n] * np.cos(np.pi*k*(2 * n + 1) / (2*N))
            y_columns[k, column] *= 2
        # -------------------------------
    
    matrix_y = y_columns
    return matrix_y

# uses numpy only
@timer
def dct_2d_numpy_sequential(block: np.ndarray):
    N = block.shape[0]  # Assuming block is 8x8
    
    # Apply 1D DCT on rows
    y_rows = np.zeros_like(block, dtype=np.float32)
    for row in range(block.shape[0]):
        # Extract the row
        x = block[row, :]
        
        # For every y_k
        for k in range(y_rows.shape[1]):
            # Type-II DCT formula right here! (for one y_k entry)
            for n in range(y_rows.shape[1]):        # summation happens here
                y_rows[row, k] += x[n] * np.cos(np.pi*k*(2 * n + 1) / (2*N))
            y_rows[row, k] *= 2
        # -------------------------------
    
    # Apply 1D DCT on columns
    y_columns = np.zeros_like(block, dtype=np.float32)
    for column in range(block.shape[1]):
        # Apply DCT formula for each column
        x = y_rows[:, column]
        
        for k in range(N):
            # Type-II DCT formula right here!
            for n in range(N):
                y_columns[k, column] += x[n] * np.cos(np.pi*k*(2 * n + 1) / (2*N))
            y_columns[k, column] *= 2
        # -------------------------------
    
    matrix_y = y_columns
    return matrix_y

# ====================================== Sequential implementations ======================================






# warmup run (compile time)
_, _ = dct_2d_jit_sequential( np.ones((1, 1)) )
_, _ = dct_2d_jit_parallel( np.ones((1, 1)) )
_, _ = dct_cuda_parallel( np.ones((1, 1)) )
_, _ = dct_cuda_parallel_scipy( np.ones((1, 1)) )
_, _ = dct_cuda_manual( np.ones((1, 1)) )

np.random.seed(42)
block_dimensions = (300, 300)
block = np.random.rand(block_dimensions[0], block_dimensions[1]) * 255
block = block - 128
block = np.astype(block, np.int8)



runs = 1

t_jit_s = 0
t_jit_p = 0
t_s = 0
t_cp_p = 0
t_cp_p_scipy = 0
t_cp_p_manual = 0

for i in range(runs):
    result, t = dct_2d_numpy_sequential( block )
    t_s += t

    result, t = dct_2d_jit_sequential( block )
    t_jit_s += t

    result, t = dct_2d_jit_parallel( block )
    t_jit_p += t

    result, t = dct_cuda_parallel( block )
    t_cp_p += t

    result, t = dct_cuda_parallel_scipy( block )
    t_cp_p_new += t

    result, t = dct_cuda_manual( block )
    t_cp_p_manual += t

print(f"Sequential DCT time: {t_s / runs} s")
print(f"JIT Sequential DCT time: {t_jit_s / runs} s")
print(f"JIT parallel DCT time: {t_jit_p / runs} s")
print(f"CuPy parallel DCT time: {t_cp_p / runs} s")
print(f"CuPy parallel DCT scipy time: {t_cp_p_scipy / runs} s")
print(f"CuPy parallel DCT manual time: {t_cp_p_manual / runs} s")