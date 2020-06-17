import gdal
import numpy as np
import TronGisPy as tgp

def dem_to_hillshade(src_raster, band=1, alg='Horn', azimuth=315, altitude=45):
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
    return raster with no_data_value = 0
    '''
    azimuth -= 180
    options = dict(band=band, alg=alg,  azimuth=azimuth, altitude=altitude, format='MEM')
    ds_src = src_raster.to_gdal_ds()
    ds = gdal.DEMProcessing('', ds_src, 'hillshade', **options)
    dst_raster = tgp.read_gdal_ds(ds)
    return dst_raster

def dem_to_slope(src_raster, band=1, alg='Horn', slope_format='degree', no_data_value=None):
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
    return raster with no_data_value = -9999
    '''
    options = dict(band=band, alg=alg,  slopeFormat=slope_format, format='MEM')
    ds_src = src_raster.to_gdal_ds()
    ds = gdal.DEMProcessing('', ds_src, 'slope', **options)
    dst_raster = tgp.read_gdal_ds(ds)
    return dst_raster

def dem_to_aspect(src_raster, band=1, alg='Horn', trigonometric=False, no_data_value=None):
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
    return raster with no_data_value = -9999
    '''
    options = dict(band=band, alg=alg,  trigonometric=trigonometric, format='MEM')
    ds_src = src_raster.to_gdal_ds()
    ds = gdal.DEMProcessing('', ds_src, 'aspect', **options)
    dst_raster = tgp.read_gdal_ds(ds)
    return dst_raster

def dem_to_TRI(src_raster, band=1, alg='Horn', no_data_value=None):
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
    return raster with no_data_value = -9999
    '''
    options = dict(band=band, alg=alg, format='MEM')
    ds_src = src_raster.to_gdal_ds()
    ds = gdal.DEMProcessing('', ds_src, 'TRI', **options)
    dst_raster = tgp.read_gdal_ds(ds)
    return dst_raster

def dem_to_TPI(src_raster, band=1, alg='Horn', no_data_value=None):
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
    return raster with no_data_value = -9999
    '''
    options = dict(band=band, alg=alg, format='MEM')
    ds_src = src_raster.to_gdal_ds()
    ds = gdal.DEMProcessing('', ds_src, 'TPI', **options)
    dst_raster = tgp.read_gdal_ds(ds)
    return dst_raster

def dem_to_roughness(src_raster, band=1, alg='Horn', no_data_value=None):
    '''
    band:
        source band number to use
    alg:
        'ZevenbergenThorne' or 'Horn'. 
        The literature suggests Zevenbergen & Thorne to be more suited to smooth landscapes, 
        where Horn’s formula to perform better on rougher terrain.
    no_data_value:
        No data value in output tiff. If None the value will follow src dem's no data value.
    return raster with no_data_value = -9999
    '''
    options = dict(band=band, alg=alg, format='MEM')
    ds_src = src_raster.to_gdal_ds()
    ds = gdal.DEMProcessing('', ds_src, 'Roughness', **options)
    dst_raster = tgp.read_gdal_ds(ds)
    return dst_raster

#TODO
# 1. Plan Curvature, Profile Curvature
# 2. LS-Factor
#==================================
