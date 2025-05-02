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
def dct_cupy_parallel(block: np.ndarray):
    """
    Uses vectorized operations for DCT (instead of the nested for-loops in dct_cuda_parallel - old version)
    """
    block_gpu = cp.array(block, dtype=cp.float32)
    N = block_gpu.shape[0]

    # Create DCT coefficient matrix once
    k = cp.arange(N).reshape(-1, 1)
    n = cp.arange(N).reshape(1, -1)
    dct_matrix = cp.cos(cp.pi * (2*n + 1) * k / (2*N)) * 2

    # Apply row-wise DCT
    temp = dct_matrix @ block_gpu

    # Apply column-wise DCT
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



np.random.seed(42)


################## generate 7 different block sizes for runtime plotting ##################
block_sizes = [2, 4, 8, 16, 32, 64, 128]

blocks = []
for block_size in block_sizes:
    block = np.random.rand(block_size, block_size) * 255
    block = block - 128
    block = np.astype(block, np.int8)

    blocks.append(block)
################## generate 9 different block sizes for runtime plotting ##################


################## Run 2D DCT algorithms for each block `runs` times ##################

runs = 10

avg_times = {   # use indices as block size references AKA 0th index is blocksize 2x2 etc
    "numpy_seq": [],
    "jit_seq": [],
    "jit_par": [],
    "cupy_par": [],
}
for block in blocks:
    print(f"\n [+] Running for block shape {block.shape}")

    t_jit_s = 0
    t_jit_p = 0
    t_s = 0
    t_cp_p = 0

    # warmup run (compile time)
    _, _ = dct_2d_jit_sequential( block )
    _, _ = dct_2d_jit_parallel( block )
    _, _ = dct_cupy_parallel( block )
    
    for i in range(runs):
        print(f"\tRun {i + 1} / {runs}")
        result, t = dct_2d_numpy_sequential( block )
        t_s += t

        result, t = dct_2d_jit_sequential( block )
        t_jit_s += t

        result, t = dct_2d_jit_parallel( block )
        t_jit_p += t

        result, t = dct_cupy_parallel( block )
        t_cp_p += t

    # print(f"\tSequential DCT time: {t_s / runs} s")
    avg_times["numpy_seq"].append(t_s / runs)

    # print(f"\tJIT Sequential DCT time: {t_jit_s / runs} s")
    avg_times["jit_seq"].append(t_jit_s / runs)
    # print(f"\tJIT parallel DCT time: {t_jit_p / runs} s")
    avg_times["jit_par"].append(t_jit_p / runs)

    # print(f"\tCuPy parallel DCT time: {t_cp_p / runs} s")
    avg_times["cupy_par"].append(t_cp_p / runs)


################## Run 2D DCT algorithms for each block `runs` times ##################



# plotting

for key in avg_times:
    # if key == "numpy_seq" or key == "jit_seq":
    #     continue
    plt.plot(block_sizes, np.array(avg_times[key])*1000, label=key)
    print("x axis", block_sizes)
    print("y axis", avg_times[key])

plt.title("2D DCT algorithms comparison")
plt.xlabel("Block size")
plt.ylabel("Time (ms)")
plt.ylim(0, max(avg_times["jit_seq"]) * 1.1 * 1000)
plt.legend()
plt.grid()
plt.show()
