import os
import gdal
import numpy as np
from numba import jit
import TronGisPy as tgp
from scipy.interpolate import griddata


def __band_interpolation(data, method='linear', no_data_value=None):
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
    use scipy.interpolate.griddata engine. Note: Use majority_interpolation, mean_interpolation
    and gdal_fillnodata to speed up.

    Parameters
    ----------
    data: array_like
        The digital number of the image which is in (n_rows, n_cols, n_bands) shape.
    method: {'nearest', 'linear', 'cubic'}, optional, default: linear
        use scipy.interpolate.griddata function.
    no_data_value: int or float, optional, default: None
        The vealue to be filled with interpolated value. If no_data_value == None, use np.nan as no_data_value.

    Returns
    -------
    data_interp: ndarray. 
        Interpolation result.

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
        data_interp[:,:,b] = __band_interpolation(data_interp[:,:,b], method='linear', no_data_value=None)
    return data_interp

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __jit_majority_interpolation(X_stacked, no_data_value): # Function is compiled to machine code when called the first time
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
def __jit_mean_interpolation(X_stacked, no_data_value): # Function is compiled to machine code when called the first time
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

def __generate_stacked_X(X_padded, ws):
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

def __majority_interpolation_single(X, no_data_value, window_size):
    X_padded = np.pad(X, ((1, 1), (1, 1)), mode='edge')
    X_stacked = __generate_stacked_X(X_padded, window_size)
    X_interp = __jit_majority_interpolation(X_stacked, no_data_value=no_data_value)
    return X_interp

def __mean_interpolation_single(X, no_data_value, window_size):
    X_padded = np.pad(X, ((1, 1), (1, 1)), mode='edge')
    X_stacked = __generate_stacked_X(X_padded, window_size)
    X_interp = __jit_mean_interpolation(X_stacked, no_data_value=no_data_value)
    return X_interp

def majority_interpolation(data, no_data_value=999, window_size=3, loop_to_fill_all=True, loop_limit=5):
    """Interpolate values on specific cells (generally nan cell) using 
    the majority value in the window.

    Parameters
    ----------
    data: array_like
        The digital number of the image which is in (n_rows, n_cols) shape. 
        The dtype of the array should be integer in order to calculate the majority.
    no_data_value: int or float, optional, default: 999
        The value to be filled with interpolated value. If no_data_value == None, 
        use np.nan as no_data_value.
    window_size: int, optional, default: 3
        The size of the window of the convolution to calculate the majority value. 
        Window_size should be odd number.
    loop_to_fill_all: bool, optional, default: True
        Fill all no_data_value until there is no no_data_value value in the data.
    loop_limit: int, optional, default: 5
        The maximum limitation on loop. if loop_to_fill_all==True, loop_limit will 
        be considered. `-1` means no limitation.

    Returns
    -------
    data_interp: ndarray. 
        Interpolation result.

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
    assert window_size%2==1 , "window_size should be odd number"
    data_interp = __majority_interpolation_single(data, no_data_value=no_data_value, window_size=window_size)
    if loop_to_fill_all and (loop_limit != -1):
        loop_count = 0
        while (np.sum(data_interp==no_data_value) > 0) and loop_count<loop_limit:
            data_interp = __majority_interpolation_single(data_interp, no_data_value=no_data_value, window_size=window_size)
            loop_count += 1    
    elif loop_to_fill_all and (loop_limit == -1):
        while np.sum(data_interp==no_data_value) > 0:
            data_interp = __majority_interpolation_single(data_interp, no_data_value=no_data_value, window_size=window_size)
    return data_interp

def mean_interpolation(data, no_data_value=999, window_size=3, loop_to_fill_all=True, loop_limit=5):
    """Interpolate values on specific cells (no_data_value) using 
    the mean value in the window.

    Parameters
    ----------
    data: array_like. 
        The digital number of the image which is in (n_rows, n_cols) shape. 
        The dtype of the array should be integer in order to calculate the majority.
    no_data_value: int or float, optional, default: 999
        The value to be filled with interpolated value. If no_data_value == None, 
        use np.nan as no_data_value.
    window_size: int, optional, default: 3
        The size of the window of the convolution to calculate the majority value. 
        Window_size should be odd number.
    loop_to_fill_all: bool, optional, default: True
        Fill all no_data_value until there is no no_data_value value in the data.
    loop_limit: int, optional, default: 5
        The maximum limitation on loop. if loop_to_fill_all==True, loop_limit will 
        be considered. `-1` means no limitation.

    Returns
    -------
    data_interp: ndarray. 
        Interpolation result.

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
    assert window_size%2==1 , "window_size should be odd number"
    data_interp = __mean_interpolation_single(data, no_data_value=no_data_value, window_size=window_size)
    if loop_to_fill_all and (loop_limit != -1): # with loop_limit
        loop_count = 0
        while (np.sum(data_interp==no_data_value) > 0) and loop_count<loop_limit:
            data_interp = __mean_interpolation_single(data_interp, no_data_value=no_data_value, window_size=window_size)
            loop_count += 1    
    elif loop_to_fill_all and (loop_limit == -1): # without loop_limit
        while np.sum(data_interp==no_data_value) > 0:
            data_interp = __mean_interpolation_single(data_interp, no_data_value=no_data_value, window_size=window_size)
    return data_interp

def gdal_fillnodata(raster, band=0, no_data_value=999, max_distance=100, smoothing_iterations=0):
    """Interpolate values on specific cells (generally nan cell) using 
    `gdal.FillNodata`. To be mentioned, this cannot accept zero in its data
    except its no_data_value is zero.

    Parameters
    ----------
    raster: Raster
        The Raster object you want to fill the no_data_value.
    band: int, optional, default: 0
        The band numnber of Raster object you want to fill the no_data_value 
        which start from zero.
    no_data_value: int or float, optional, default: 999
        The value to be filled with interpolated value. If no_data_value == None, 
        use np.nan as no_data_value.
    max_distance: int, optional, default: 100
        The cells within the max distance from no_data_value location will be 
        calculated to fill the no_data_value.
    smoothing_iterations: int, optional, default: 0
        The maximum limitation on loop.

    Returns
    -------
    data_interp: ndarray. 
        Interpolation result.

    Examples
    -------- 
    >>> import numpy as np
    >>> import TronGisPy as tgp 
    >>> from TronGisPy import Interpolation
    >>> from matplotlib import pyplot as plt
    >>> raster_fp = tgp.get_testing_fp('tif_forinterpolation')
    >>> raster = tgp.read_raster(raster_fp)
    >>> raster.data[np.isnan(raster.data)] = 999
    >>> raster_interp = Interpolation.gdal_fillnodata(raster)
    >>> fig, (ax1, ax2) = plt.subplots(1,2)
    >>> raster.plot(ax=ax1)
    >>> raster_interp.plot(ax=ax2)
    >>> plt.show()
    """    
    # make dst_band
    ds_dst = raster.to_gdal_ds()
    dstband = ds_dst.GetRasterBand(band+1)

    # make maskband
    raster_mask = raster.copy()
    raster_mask.data[raster_mask.data==no_data_value] = 0
    raster_mask.data[raster_mask.data!=0] = 1
    raster_mask.astype(np.bool)
    raster_mask.no_data_value = 0
    ds_mask = raster_mask.to_gdal_ds()
    maskband = ds_mask.GetRasterBand(1)

    gdal.FillNodata(dstband, maskband, max_distance, smoothing_iterations)
    data_interp = tgp.read_gdal_ds(ds_dst)

    ds_dst = None
    ds_mask = None
    return data_interp
