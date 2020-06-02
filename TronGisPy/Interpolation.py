import os
import numpy as np
from numba import jit
from scipy.interpolate import griddata



def band_interpolation(X, method='linear', mask=None):
    """
    X: please whole image.
    method: 'nearest', 'linear', 'cubic', see scipy.interpolate.griddata documentation.
    mask: The location to be filled with value. If mask == None, use np.nan(X) as mask.
    """
    assert len(X.shape) == 2, "X should be one band image!"
    if mask is not None:
        assert X.shape == mask.shape, 'X.shape should be mask.shape!'
        assert mask.dtype == np.bool, 'mask.dtype should be np.bool!'
        points = np.array(np.where(~mask)).T
        values = X[~mask].copy()
    else:
        points = np.array(np.where(~np.isnan(X))).T
        values = X[~np.isnan(X)].copy()
    grid_x, grid_y = np.where(np.ones_like(X))
    grid_x, grid_y = grid_x.reshape(X.shape), grid_y.reshape(X.shape)
    X_interp = griddata(points, values, (grid_x, grid_y), method=method)
    return X_interp

def img_interpolation(X, method='linear', mask=None):
    """
    X: please whole image.
    method: 'nearest', 'linear', 'cubic', see scipy.interpolate.griddata documentation.
    mask: The location to be filled with value. If mask == None, use np.nan(X) as mask.
    """
    X_interp = X.copy()
    if len(X_interp.shape) == 2:
        X_interp = np.expand_dims(X_interp, axis=2)
    for b in range(X_interp.shape[2]):
        X_interp[:,:,b] = band_interpolation(X_interp[:,:,b], method='linear', mask=None)
    return X_interp



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

def majority_interpolation(X, no_data_value=999, window_size=3, loop_to_fill_all=True, loop_limit=-1):
    """
    X: the image with no_data_value, should be in np.int type
    no_data_value: where you want to fill in the value, should not be np.nan or negative value.
    window_size: the window size of the convolution to calculate the majority.
    """
    assert len(X.shape) == 2, "X should have onle 2 dimension"
    assert np.issubdtype(X.dtype, np.integer), "X should be in integer type"
    X_interp = _majority_interpolation_single(X, no_data_value=no_data_value, window_size=window_size)
    if loop_to_fill_all and (loop_limit != -1):
        loop_count = 0
        while (np.sum(X_interp==no_data_value) > 0) and loop_count<loop_limit:
            X_interp = _majority_interpolation_single(X_interp, no_data_value=no_data_value, window_size=window_size)
            loop_count += 1    
    elif loop_to_fill_all and (loop_limit == -1):
        while np.sum(X_interp==no_data_value) > 0:
            X_interp = _majority_interpolation_single(X_interp, no_data_value=no_data_value, window_size=window_size)
    return X_interp

def mean_interpolation(X, no_data_value=999, window_size=3, loop_to_fill_all=True, loop_limit=-1):
    """
    X: the image with no_data_value, should be in np.int type
    no_data_value: where you want to fill in the value, should not be np.nan or negative value.
    window_size: the window size of the convolution to calculate the majority.
    """
    assert len(X.shape) == 2, "X should have onle 2 dimension"
    X_interp = _mean_interpolation_single(X, no_data_value=no_data_value, window_size=window_size)
    if loop_to_fill_all and (loop_limit != -1): # with loop_limit
        loop_count = 0
        while (np.sum(X_interp==no_data_value) > 0) and loop_count<loop_limit:
            X_interp = _mean_interpolation_single(X_interp, no_data_value=no_data_value, window_size=window_size)
            loop_count += 1    
    elif loop_to_fill_all and (loop_limit == -1): # without loop_limit
        while np.sum(X_interp==no_data_value) > 0:
            X_interp = _mean_interpolation_single(X_interp, no_data_value=no_data_value, window_size=window_size)
    return X_interp