import numpy as np
import numba


@numba.jit("f8[:,:](f8[:], f8[:])", cache=True, nopython=True)
def calc_dist_matrix_1d_jit(loc_larger: np.ndarray, loc_smaller: np.ndarray) -> np.ndarray:
    mat_size = (len(loc_larger), len(loc_smaller))
    _loc_smaller_mat = np.ones(mat_size, dtype=np.float64) * loc_smaller
    _loc_larger_mat = np.ones(mat_size, dtype=np.float64) * loc_larger[:, np.newaxis]
    return np.abs(_loc_larger_mat - _loc_smaller_mat)


@numba.jit("f8[:,:](f8[:,:], f8[:,:])", cache=True, nopython=True)
def calc_dist_matrix_2d_jit(loc_larger: np.ndarray, loc_smaller: np.ndarray) -> np.ndarray:
    mat_size = (len(loc_larger), len(loc_smaller), len(loc_smaller[0]))
    _loc_smaller_mat = np.ones(mat_size, dtype=np.float64) * loc_smaller
    _loc_larger_mat = np.ones(mat_size, dtype=np.float64) * loc_larger[:, np.newaxis]
    return np.sqrt(np.sum(np.square(_loc_larger_mat - _loc_smaller_mat), axis=2))


@numba.jit("Tuple((i8, f8))(f8[:])", cache=True, nopython=True)
def argmin_1d_jit(arr: np.ndarray) -> tuple[int, float]:
    min_val = 1.79769313486231e+308
    min_idx = -1
    for i in range(len(arr)):
        if arr[i] < min_val:
            min_val = arr[i]
            min_idx = i
    return min_idx, min_val


@numba.jit("Tuple((i8, i8))(f8[:, :])", cache=True, nopython=True, parallel=True)
def argmin_2d_jit(arr: np.ndarray) -> tuple[int, int]:
    row_nums = len(arr)
    min_col_indices = np.zeros(row_nums, dtype=np.int64)
    min_values = np.zeros(row_nums, dtype=np.float64)
    for irow in numba.prange(row_nums):
        min_col_indices[irow], min_values[irow] = argmin_1d_jit(arr[irow])

    min_idx, min_val = argmin_1d_jit(min_values)

    return min_idx, min_col_indices[min_idx]


@numba.jit("f8[:](f8[:, :])", cache=True, nopython=True)
def find_proximal_jit(dist_matrix: np.ndarray) -> np.ndarray:
    max_val = 1.79769313486231e+308

    h_size = len(dist_matrix[0])
    proximal_distance = np.zeros(h_size, dtype=np.float64)
    for _ in range(h_size):
        # find the most proximal
        min_index = argmin_2d_jit(dist_matrix)
        min_row = min_index[0]
        min_col = min_index[1]

        # log distance
        proximal_distance[min_col] = dist_matrix[min_row, min_col]

        # post process
        dist_matrix[:, min_col] = max_val

    return proximal_distance


if __name__ == "__main__":
    arr_small = np.random.random((4, 2)).astype(np.float64)
    arr_large = np.random.random((6, 2)).astype(np.float64)

    ret1 = calc_dist_matrix_1d_jit(arr_large[:, 0], arr_small[:, 0])
    print(ret1)
    ret2 = calc_dist_matrix_2d_jit(arr_large, arr_small)
    print(ret2)
    ret3 = find_proximal_jit(ret2)
    print(ret3)

    argmin_ret_1d = argmin_1d_jit(np.random.random(8))
    print(argmin_ret_1d)

    argmin_ret = argmin_2d_jit(np.random.random((10, 3)))
    print(argmin_ret)
