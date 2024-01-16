import time
import numpy as np
import scipy.sparse
import sksparse.cholmod

def benchmark_scipy_sparse(matrix):
    start_time = time.time()
    result = scipy.sparse.linalg.expm(matrix)
    elapsed_time = time.time() - start_time
    return result, elapsed_time

def benchmark_sckit_sparse(matrix):
    start_time = time.time()
    factor = sksparse.cholmod.cholesky(matrix)
    result = factor.expm1()
    elapsed_time = time.time() - start_time
    return result, elapsed_time

# Replace this with your actual sparse matrix
# For demonstration, I'm creating a random sparse matrix
matrix_size = 1000
sparse_matrix = scipy.sparse.identity(matrix_size)

# Benchmark SciPy's expm
result_scipy, time_scipy = benchmark_scipy_sparse(sparse_matrix)

# Benchmark scikit-sparse
result_sckit, time_sckit = benchmark_sckit_sparse(sparse_matrix)

print(f"SciPy elapsed time: {time_scipy:.6f} seconds")
print(f"scikit-sparse elapsed time: {time_sckit:.6f} seconds")

# You can compare the results if needed
# print(np.allclose(result_scipy, result_sckit))
