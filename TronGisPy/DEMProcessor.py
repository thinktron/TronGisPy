import numpy as np
import gdal
from TronGisPy.GisIO import get_geo_info, write_output_tif

def dem_to_hillshade(src_tif_path, dst_tif_path, band=1, alg='Horn', azimuth=315, altitude=45, no_data_value=None):
    '''
    band:
        source band number to use
    alg:
        'ZevenbergenThorne' or 'Horn'. 
        The literature suggests Zevenbergen & Thorne to be more suited to smooth landscapes, 
        where Horn’s formula to perform better on rougher terrain.
    azimuth:
        azimuth of the light, in degrees. 0 if it comes from the top of the raster, 90 from the east, ... 
        The default value, 315, should rarely be changed as it is the value generally used to generate shaded maps.
    altitude: 
        altitude of the light, in degrees. 90 if the light comes from above the DEM, 0 if it is raking light.
    no_data_value:
        No data value in output tiff. If None the value will follow src dem's no data value.
    '''
    azimuth -= 180
    options = dict(
        band = band,
        alg = alg, 
        azimuth = azimuth,
        altitude = altitude,
        format = 'MEM'
    )

    ds = gdal.DEMProcessing('', src_tif_path, 'hillshade', **options)
    band = ds.GetRasterBand(1)
    out_arr = band.ReadAsArray()
    ds = None
    
    src_geo_info = get_geo_info(src_tif_path)
    cols = src_geo_info[0]
    rows = src_geo_info[1]
    geo_transform = src_geo_info[3]
    projection = src_geo_info[4]
    gdaldtype = src_geo_info[5]
    no_data_value = src_geo_info[-1] if no_data_value is None else no_data_value
    
    out_arr = np.where(out_arr==0, np.nan, out_arr)
    write_output_tif(out_arr, dst_tif_path, bands=1, cols=cols, rows=rows, geo_transform=geo_transform, projection=projection,
                           gdaldtype=gdaldtype, no_data_value=no_data_value)

def dem_to_slope(src_tif_path, dst_tif_path, band=1, alg='Horn', slope_format='degree', no_data_value=None):
    '''
    band:
        source band number to use
    alg:
        'ZevenbergenThorne' or 'Horn'. 
        The literature suggests Zevenbergen & Thorne to be more suited to smooth landscapes, 
        where Horn’s formula to perform better on rougher terrain.
    slopeformat:
        "degree" or "percent".
    no_data_value:
        No data value in output tiff. If None the value will follow src dem's no data value.
    '''
    options = dict(
        band = band,
        alg = alg, 
        slopeFormat = slope_format,
        format = 'MEM'
    )

    ds = gdal.DEMProcessing('', src_tif_path, 'slope', **options)
    band = ds.GetRasterBand(1)
    out_arr = band.ReadAsArray()
    ds = None
    
    src_geo_info = get_geo_info(src_tif_path)
    cols = src_geo_info[0]
    rows = src_geo_info[1]
    geo_transform = src_geo_info[3]
    projection = src_geo_info[4]
    gdaldtype = src_geo_info[5]
    no_data_value = src_geo_info[-1] if no_data_value is None else no_data_value
    
    out_arr = np.where(out_arr==-9999, np.nan, out_arr)
    write_output_tif(out_arr, dst_tif_path, bands=1, cols=cols, rows=rows, geo_transform=geo_transform, projection=projection,
                           gdaldtype=gdaldtype, no_data_value=no_data_value)

def dem_to_aspect(src_tif_path, dst_tif_path, band=1, alg='Horn', trigonometric=False, no_data_value=None):
    '''
    band:
        source band number to use
    alg:
        'ZevenbergenThorne' or 'Horn'. 
        The literature suggests Zevenbergen & Thorne to be more suited to smooth landscapes, 
        where Horn’s formula to perform better on rougher terrain.
    trigonometric:
        whether to return trigonometric angle instead of azimuth. 
        Thus 0deg means East, 90deg North, 180deg West, 270deg South.
    no_data_value:
        No data value in output tiff. If None the value will follow src dem's no data value.
    '''
    options = dict(
        band = band,
        alg = alg, 
        trigonometric = trigonometric,
        format = 'MEM'
    )

    ds = gdal.DEMProcessing('', src_tif_path, 'aspect', **options)
    band = ds.GetRasterBand(1)
    out_arr = band.ReadAsArray()
    ds = None
    
    src_geo_info = get_geo_info(src_tif_path)
    cols = src_geo_info[0]
    rows = src_geo_info[1]
    geo_transform = src_geo_info[3]
    projection = src_geo_info[4]
    gdaldtype = src_geo_info[5]
    no_data_value = src_geo_info[-1] if no_data_value is None else no_data_value

    out_arr = np.where(out_arr==-9999, np.nan, out_arr)
    write_output_tif(out_arr, dst_tif_path, bands=1, cols=cols, rows=rows, geo_transform=geo_transform, projection=projection,
                           gdaldtype=gdaldtype, no_data_value=no_data_value)
    
def dem_to_TRI(src_tif_path, dst_tif_path, band=1, alg='Horn', no_data_value=None):
    '''
    Terrain Ruggedness Index
    band:
        source band number to use
    alg:
        'ZevenbergenThorne' or 'Horn'. 
        The literature suggests Zevenbergen & Thorne to be more suited to smooth landscapes, 
        where Horn’s formula to perform better on rougher terrain.
    no_data_value:
        No data value in output tiff. If None the value will follow src dem's no data value.
    '''
    options = dict(
        band = band,
        alg = alg, 
        format = 'MEM'
    )

    ds = gdal.DEMProcessing('', src_tif_path, 'TRI', **options)
    band = ds.GetRasterBand(1)
    out_arr = band.ReadAsArray()
    ds = None
    
    src_geo_info = get_geo_info(src_tif_path)
    cols = src_geo_info[0]
    rows = src_geo_info[1]
    geo_transform = src_geo_info[3]
    projection = src_geo_info[4]
    gdaldtype = src_geo_info[5]
    no_data_value = src_geo_info[-1] if no_data_value is None else no_data_value
    
    out_arr = np.where(out_arr==-9999, np.nan, out_arr)
    write_output_tif(out_arr, dst_tif_path, bands=1, cols=cols, rows=rows, geo_transform=geo_transform, projection=projection,
                           gdaldtype=gdaldtype, no_data_value=no_data_value)

def dem_to_TPI(src_tif_path, dst_tif_path, band=1, alg='Horn', no_data_value=None):
    '''
    Topographic Position Index
    band:
        source band number to use
    alg:
        'ZevenbergenThorne' or 'Horn'. 
        The literature suggests Zevenbergen & Thorne to be more suited to smooth landscapes, 
        where Horn’s formula to perform better on rougher terrain.
    no_data_value:
        No data value in output tiff. If None the value will follow src dem's no data value.
    '''
    options = dict(
        band = band,
        alg = alg, 
        format = 'MEM'
    )

    ds = gdal.DEMProcessing('', src_tif_path, 'TPI', **options)
    band = ds.GetRasterBand(1)
    out_arr = band.ReadAsArray()
    ds = None
    
    src_geo_info = get_geo_info(src_tif_path)
    cols = src_geo_info[0]
    rows = src_geo_info[1]
    geo_transform = src_geo_info[3]
    projection = src_geo_info[4]
    gdaldtype = src_geo_info[5]
    no_data_value = src_geo_info[-1] if no_data_value is None else no_data_value
    
    out_arr = np.where(out_arr==-9999, np.nan, out_arr)
    write_output_tif(out_arr, dst_tif_path, bands=1, cols=cols, rows=rows, geo_transform=geo_transform, projection=projection,
                           gdaldtype=gdaldtype, no_data_value=no_data_value)

def dem_to_roughness(src_tif_path, dst_tif_path, band=1, alg='Horn', no_data_value=None):
    '''
    band:
        source band number to use
    alg:
        'ZevenbergenThorne' or 'Horn'. 
        The literature suggests Zevenbergen & Thorne to be more suited to smooth landscapes, 
        where Horn’s formula to perform better on rougher terrain.
    no_data_value:
        No data value in output tiff. If None the value will follow src dem's no data value.
    '''
    options = dict(
        band = band,
        alg = alg, 
        format = 'MEM'
    )

    ds = gdal.DEMProcessing('', src_tif_path, 'Roughness', **options)
    band = ds.GetRasterBand(1)
    out_arr = band.ReadAsArray()
    ds = None
    
    src_geo_info = get_geo_info(src_tif_path)
    cols = src_geo_info[0]
    rows = src_geo_info[1]
    geo_transform = src_geo_info[3]
    projection = src_geo_info[4]
    gdaldtype = src_geo_info[5]
    no_data_value = src_geo_info[-1] if no_data_value is None else no_data_value
    
    out_arr = np.where(out_arr==-9999, np.nan, out_arr)
    write_output_tif(out_arr, dst_tif_path, bands=1, cols=cols, rows=rows, geo_transform=geo_transform, projection=projection,
                           gdaldtype=gdaldtype, no_data_value=no_data_value)

#TODO
# 1. Plan Curvature, Profile Curvature
# 2. LS-Factor
#==================================
