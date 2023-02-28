import logging
import time
import os

import numpy as np
import numba as nb
import xarray as xr


CLASS_0: int = 0
CLASS_1: int = 1
CLASS_2: int = 2
CLASS_0_ALIAS: int = 1
CLASS_1_ALIAS: int = 2
CLASS_2_ALIAS: int = 4
HOST_DTYPE: np.dtype = np.uint8
LAYER_AXIS: int = 2
LAYER_COUNT: int = 3
CLASS_COUNT: int = 3
Y_AXIS: int = 0
X_AXIS: int = 1
SUM_NO_MODE: int = 0
NO_DATA: int = 10


# --------------------------------------------------------------------------
# SKELETON FUNCTION PT. 5
# Change function name to fit alg
# Change "alg" out with whatever you're doin
# --------------------------------------------------------------------------
@nb.njit
def _alg_product(input_array: np.ndarray) -> int:
    """
    reduction algorithm, takes 3d input, returns scalar
    """
    output_scalar = 0
    return output_scalar


# --------------------------------------------------------------------------
# SKELETON FUNCTION PT. 4
# Change function name to fit alg
# Change "alg" out with whatever you're doing
# --------------------------------------------------------------------------
@nb.njit(parallel=True)
def _fast_iterate_alg(
        output_array: np.ndarray, input_array: np.ndarray) -> np.ndarray:
    """
    Iterate through first two dims of 3d host array
    to get the mode for the z axis.
    """
    for y in nb.prange(input_array.shape[Y_AXIS]):
        for x in nb.prange(input_array.shape[X_AXIS]):
            output_array[y, x] = _alg_product(input_array[y, x, :])
    return output_array


# --------------------------------------------------------------------------
# SKELETON FUNCTION PT. 3
# Change function name to fit alg
# Change "alg" out with whatever you're doing
# --------------------------------------------------------------------------
def calculate_alg(grid_cell_data_array: xr.DataArray,
                  from_disk: bool = False,
                  grid_cell_zarr_path: str = None,
                  logger: logging.Logger = None) -> np.ndarray:
    """
    Reduction algorithm, skeleton function
    """
    if from_disk:
        if not os.path.exists(grid_cell_zarr_path):
            msg = f'{grid_cell_zarr_path} does ' + \
                'cannot be found or does not exist.'
            raise FileNotFoundError(msg)
        grid_cell_data_array = xr.from_zarr(grid_cell_zarr_path)
    grid_cell_shape = grid_cell_data_array.shape
    output_array = np.zeros(
        (grid_cell_shape[Y_AXIS], grid_cell_shape[X_AXIS]), dtype=HOST_DTYPE)
    logger.info('Computing alg')
    st = time.time()
    output_array = _fast_iterate_alg(output_array, grid_cell_data_array)
    et = time.time()
    logger.info('Alg compute time {}'.format(round(et-st, 3)))
    return output_array


@nb.njit
def _mode_sum_product(array: np.ndarray) -> int:
    """
    Multi-modal function
    Given a single dimension host array where each index is a class
    return all occurences of the max in the array such that if
    multiple classes have the same max, return the sum.
    :param array: np.ndarray, Flat array to calculate multi-mode
    :return max_element_to_return: int, element
    """
    max_element = np.max(array)
    if max_element == SUM_NO_MODE:
        return NO_DATA
    max_indices = np.argwhere(array == max_element).flatten()
    max_pl = np.where(max_indices == CLASS_0, CLASS_0_ALIAS, max_indices)
    max_pl = np.where(max_indices == CLASS_1, CLASS_1_ALIAS, max_pl)
    max_pl = np.where(max_indices == CLASS_2, CLASS_2_ALIAS, max_pl)
    max_element_to_return = int(np.sum(max_pl))
    return max_element_to_return


@nb.njit(parallel=True)
def _fast_iterate_mode(mode: np.ndarray, array: np.ndarray) -> np.ndarray:
    """
    Iterate through first two dims of 3d host array
    to get the mode for the z axis.
    """
    for y in nb.prange(array.shape[Y_AXIS]):
        for x in nb.prange(array.shape[X_AXIS]):
            mode[y, x] = _mode_sum_product(array[y, x, :])
    return mode


def _get_sum(binary_class: str,
             logger: logging.Logger) -> np.ndarray:
    """
    Given a binary class occurance ndarray, sum along the z
    axis. Converts zarr dask array to device-backed dask array,
    performs computation, converts back to host array.
    """
    st = time.time()
    sum_class = binary_class.sum(axis=LAYER_AXIS, dtype=HOST_DTYPE)
    et = time.time()
    logger.info('Mode sum - Sum time: {}'.format(et-st))
    return sum_class


def calculate_mode(grid_cell_data_array: xr.DataArray,
                   classes: dict,
                   from_disk: bool = False,
                   grid_cell_zarr_path: str = None,
                   logger: logging.Logger = None) -> np.ndarray:
    """
    Get the mode from a series of binary class occurance arrays.
    """
    num_classes = len(classes.keys())
    if from_disk:
        if not os.path.exists(grid_cell_zarr_path):
            msg = f'{grid_cell_zarr_path} does ' + \
                'cannot be found or does not exist.'
            raise FileNotFoundError(msg)
        grid_cell_data_array = xr.from_zarr(grid_cell_zarr_path)
    grid_cell_shape = grid_cell_data_array.shape
    class_sums_shape = (grid_cell_shape[Y_AXIS], grid_cell_shape[X_AXIS],
                        num_classes)
    class_sums = np.zeros(class_sums_shape, dtype=HOST_DTYPE)
    for class_id, class_value in classes.items():
        class_binary_array = xr.where(
            grid_cell_data_array == class_value, 1, 0).astype(HOST_DTYPE)
        class_sums[:, :, class_id] = _get_sum(class_binary_array, logger).data
    mode = np.zeros(
        (class_sums.shape[Y_AXIS], class_sums.shape[X_AXIS]), dtype=HOST_DTYPE)
    logger.info('Computing mode')
    st = time.time()
    mode = _fast_iterate_mode(mode, class_sums)
    et = time.time()
    logger.info('Mode compute time {}'.format(round(et-st, 3)))
    return mode
