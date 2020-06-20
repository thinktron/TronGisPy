import os
import numpy as np
from numba import jit
from scipy.interpolate import griddata



def band_interpolation(data, method='linear', no_data_value=None):   
    """be used in img_interpolation"""
    assert len(data.shape) == 2, "data should be one band image!"
    if no_data_value is not None:
        data[data==no_data_value] = np.nan
    points = np.array(np.where(~np.isnan(data))).T
    values = data[~np.isnan(data)].copy()
    grid_x, grid_y = np.where(np.ones_like(data))
    grid_x, grid_y = grid_x.reshape(data.shape), grid_y.reshape(data.shape)
    X_interp = griddata(points, values, (grid_x, grid_y), method=method)
    return X_interp

def img_interpolation(data, method='linear', no_data_value=None):
    """Interpolate values on specific cells (generally nan cell) of the image data.

    Parameters
    ----------
    data: `numpy.array`. The digital number of the image which is in 
    (n_rows, n_cols, n_bands) shape.

    method: str. Should be in {'nearest', 'linear', 'cubic'}, see also 
    scipy.interpolate.griddata documentation.

    no_data_value: int or float. The vealue to be filled with interpolated value. If 
    no_data_value == None, use np.nan as no_data_value.

    Returns
    -------
    data_interp: `numpy.array`. Interpolation result.

    Examples
    -------- 
    >>> from matplotlib import pyplot as plt
    >>> import TronGisPy as tgp 
    >>> from TronGisPy import Interpolation
    >>> raster_fp = tgp.get_testing_fp('tif_forinterpolation')
    >>> raster = tgp.read_raster(raster_fp)
    >>> data_interp = Interpolation.img_interpolation(raster.data)
    >>> fig, (ax1, ax2) = plt.subplots(1,2)
    >>> ax1.imshow(raster.data[:, :, 0])
    >>> ax2.imshow(data_interp[:, :, 0])
    >>> plt.show()
    """
    data_interp = data.copy()
    if len(data_interp.shape) == 2:
        data_interp = np.expand_dims(data_interp, axis=2)
    for b in range(data_interp.shape[2]):
        data_interp[:,:,b] = band_interpolation(data_interp[:,:,b], method='linear', no_data_value=None)
    return data_interp



@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def _jit_majority_interpolation(X_stacked, no_data_value): # Function is compiled to machine code when called the first time
    X_interp = np.zeros((X_stacked.shape[0], X_stacked.shape[1]), dtype=np.int32)
    for i in range(X_interp.shape[0]-1):
        for j in range(X_interp.shape[1]-1):
            if X_stacked[i,j][5] == no_data_value: # only calculate majority when value is no_data_value
                vals = X_stacked[i,j][X_stacked[i,j]!=no_data_value]
                v_len = vals.shape[0]
                if v_len !=0: # if at least one value in the convolution is not no_data_value
                    X_interp[i, j] = np.bincount(vals).argmax() # calculate majority 
                else: 
                    X_interp[i, j] = no_data_value
            else:
                X_interp[i, j] = X_stacked[i,j][5]
    return X_interp

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def _jit_mean_interpolation(X_stacked, no_data_value): # Function is compiled to machine code when called the first time
    X_interp = np.zeros((X_stacked.shape[0], X_stacked.shape[1]))
    for i in range(X_interp.shape[0]-1):
        for j in range(X_interp.shape[1]-1):
            if X_stacked[i,j][5] == no_data_value: # only calculate majority when value is no_data_value
                vals = X_stacked[i,j][X_stacked[i,j]!=no_data_value]
                v_len = vals.shape[0]
                if v_len !=0: # if at least one value in the convolution is not no_data_value
                    X_interp[i, j] = np.mean(vals) # calculate mean
                else: 
                    X_interp[i, j] = no_data_value
            else:
                X_interp[i, j] = X_stacked[i,j][5]
    return X_interp

def _generate_stacked_X(X_padded, ws):
    """
    X_stacked = np.stack([
        X_padded[0:-2, 0:-2], X_padded[1:-1, 0:-2], X_padded[2:, 0:-2],
        X_padded[0:-2, 1:-1], X_padded[1:-1, 1:-1], X_padded[2:, 1:-1],
        X_padded[0:-2, 2:  ], X_padded[1:-1, 2:  ], X_padded[2:, 2:  ]
        ] , axis=-1)
    """
    rows, cols = X_padded.shape
    X_stacked = []
    for i in range(ws):
        for j in range(ws):
            X_stacked.append(X_padded[i:rows-(ws-i)+1, j:cols-(ws-j)+1])
    X_stacked = np.stack(X_stacked, axis=-1)
    return X_stacked

def _majority_interpolation_single(X, no_data_value, window_size):
    X_padded = np.pad(X, ((1, 1), (1, 1)), mode='edge')
    X_stacked = _generate_stacked_X(X_padded, window_size)
    X_interp = _jit_majority_interpolation(X_stacked, no_data_value=no_data_value)
    return X_interp

def _mean_interpolation_single(X, no_data_value, window_size):
    X_padded = np.pad(X, ((1, 1), (1, 1)), mode='edge')
    X_stacked = _generate_stacked_X(X_padded, window_size)
    X_interp = _jit_mean_interpolation(X_stacked, no_data_value=no_data_value)
    return X_interp

def majority_interpolation(data, no_data_value=999, window_size=3, loop_to_fill_all=True, loop_limit=-1):
    """Interpolate values on specific cells (generally nan cell) using 
    the majority value in the window.

    Parameters
    ----------
    data: `numpy.array`. The digital number of the image which is in 
    (n_rows, n_cols) shape. The dtype of the array should be integer in 
    order to calculate the majority.

    no_data_value: int. The value to be filled with interpolated value. If 
    no_data_value == None, use np.nan as no_data_value.

    window_size: int. The size of the window of the convolution to calculate 
    the majority value. window_size should be odd number.

    loop_to_fill_all: bool. Fill all no_data_value until there is no no_data_value 
    value in the data.

    loop_limit: bool. The maximum limitation on loop. if loop_to_fill_all==True, 
    loop_limit will be considered.

    Returns
    -------
    data_interp: `numpy.array`. Interpolation result.

    Examples
    -------- 
    >>> import numpy as np
    >>> import TronGisPy as tgp 
    >>> from TronGisPy import Interpolation
    >>> from matplotlib import pyplot as plt
    >>> raster_fp = tgp.get_testing_fp('tif_forinterpolation')
    >>> raster = tgp.read_raster(raster_fp)
    >>> data = raster.data.copy()[:, :, 0]
    >>> data_interp = data.copy()
    >>> data_interp[np.isnan(data_interp)] = 999
    >>> data_interp = data_interp.astype(np.int)
    >>> data_interp = Interpolation.majority_interpolation(data_interp, no_data_value=999)
    >>> fig, (ax1, ax2) = plt.subplots(1,2)
    >>> ax1.imshow(data)
    >>> ax2.imshow(data_interp)
    >>> plt.show()
    """
    assert len(data.shape) == 2, "data should have onle 2 dimension"
    assert np.issubdtype(data.dtype, np.integer), "data should be in integer type"
    assert no_data_value%2==1 , "no_data_value should be odd number"
    data_interp = _majority_interpolation_single(data, no_data_value=no_data_value, window_size=window_size)
    if loop_to_fill_all and (loop_limit != -1):
        loop_count = 0
        while (np.sum(data_interp==no_data_value) > 0) and loop_count<loop_limit:
            data_interp = _majority_interpolation_single(data_interp, no_data_value=no_data_value, window_size=window_size)
            loop_count += 1    
    elif loop_to_fill_all and (loop_limit == -1):
        while np.sum(data_interp==no_data_value) > 0:
            data_interp = _majority_interpolation_single(data_interp, no_data_value=no_data_value, window_size=window_size)
    return data_interp

def mean_interpolation(data, no_data_value=999, window_size=3, loop_to_fill_all=True, loop_limit=-1):
    """Interpolate values on specific cells (generally nan cell) using 
    the majority value in the window.

    Parameters
    ----------
    data: `numpy.array`. The digital number of the image which is in 
    (n_rows, n_cols) shape. The dtype of the array should be integer in 
    order to calculate the majority.

    no_data_value: int. The value to be filled with interpolated value. If 
    no_data_value == None, use np.nan as no_data_value.

    window_size: int. The size of the window of the convolution to calculate 
    the mean value. window_size should be odd number.

    loop_to_fill_all: bool. Fill all no_data_value until there is no no_data_value 
    value in the data.

    loop_limit: bool. The maximum limitation on loop. if loop_to_fill_all==True, 
    loop_limit will be considered.

    Returns
    -------
    data_interp: `numpy.array`. Interpolation result.

    Examples
    -------- 
    >>> import numpy as np
    >>> import TronGisPy as tgp 
    >>> from TronGisPy import Interpolation
    >>> from matplotlib import pyplot as plt
    >>> raster_fp = tgp.get_testing_fp('tif_forinterpolation')
    >>> raster = tgp.read_raster(raster_fp)
    >>> data = raster.data.copy()[:, :, 0]
    >>> data_interp = data.copy()
    >>> data_interp[np.isnan(data_interp)] = 999
    >>> data_interp = Interpolation.mean_interpolation(data_interp, no_data_value=999)
    >>> fig, (ax1, ax2) = plt.subplots(1,2)
    >>> ax1.imshow(data)
    >>> ax2.imshow(data_interp)
    >>> plt.show()
    """
    assert len(data.shape) == 2, "data should have onle 2 dimension"
    assert no_data_value%2==1 , "no_data_value should be odd number"
    data_interp = _mean_interpolation_single(data, no_data_value=no_data_value, window_size=window_size)
    if loop_to_fill_all and (loop_limit != -1): # with loop_limit
        loop_count = 0
        while (np.sum(data_interp==no_data_value) > 0) and loop_count<loop_limit:
            data_interp = _mean_interpolation_single(data_interp, no_data_value=no_data_value, window_size=window_size)
            loop_count += 1    
    elif loop_to_fill_all and (loop_limit == -1): # without loop_limit
        while np.sum(data_interp==no_data_value) > 0:
            data_interp = _mean_interpolation_single(data_interp, no_data_value=no_data_value, window_size=window_size)
    return data_interp