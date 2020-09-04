import gdal
import numpy as np
import TronGisPy as tgp

def dem_to_hillshade(src_raster, band=0, alg='Horn', azimuth=315, altitude=45):
    """Calculate the hillshade for the DEM.

    Parameters
    ----------
    src_raster : Raster
        The dem used to calculate the hillshade.
    band : int, optional, default: 0
        source band number to use.
    alg : {'ZevenbergenThorne' or 'Horn'}, optional, default: Horn
        The literature suggests Zevenbergen & Thorne to be more suited to smooth landscapes, 
        where Horn’s formula to perform better on rougher terrain.
    azimuth : int, optional, default 315
        Azimuth of the light, in degrees. 0 if it comes from the top of the raster, 90 from the east, ... 
        The default value, 315, should rarely be changed as it is the value generally used to generate shaded maps.
    altitude : int, optional, default 45
        Altitude of the light, in degrees. 90 if the light comes from above the DEM, 0 if it is raking light.

    Returns
    -------
    dst_raster: Raster
        Hillshade calculated from the DEM.
    """
    options = dict(band=band+1, alg=alg,  azimuth=azimuth, altitude=altitude, format='MEM')
    ds_src = src_raster.to_gdal_ds()
    ds = gdal.DEMProcessing('', ds_src, 'hillshade', **options)
    dst_raster = tgp.read_gdal_ds(ds)
    return dst_raster

def dem_to_slope(src_raster, band=0, alg='Horn', slope_format='degree'):
    """Calculate the slope for the DEM.

    Parameters
    ----------
    src_raster : Raster
        The dem used to calculate the slope.
    band : int, optional, default: 0
        source band number to use.
    alg : {'ZevenbergenThorne' or 'Horn'}, optional, default: Horn
        The literature suggests Zevenbergen & Thorne to be more suited to smooth landscapes, 
        where Horn’s formula to perform better on rougher terrain.
    slope_format: {"degree" or "percent"}, optional, default degree
        The format of the slope.

    Returns
    -------
    dst_raster: Raster
        Slope calculated from the DEM.
    """
    options = dict(band=band+1, alg=alg,  slopeFormat=slope_format, format='MEM')
    ds_src = src_raster.to_gdal_ds()
    ds = gdal.DEMProcessing('', ds_src, 'slope', **options)
    dst_raster = tgp.read_gdal_ds(ds)
    return dst_raster

def dem_to_aspect(src_raster, band=0, alg='Horn', trigonometric=False):
    """Calculate the aspect for the DEM.

    Parameters
    ----------
    src_raster : Raster
        The dem used to calculate the aspect.
    band : int, optional, default: 0
        source band number to use.
    alg : {'ZevenbergenThorne' or 'Horn'}, optional, default: Horn
        The literature suggests Zevenbergen & Thorne to be more suited to smooth landscapes, 
        where Horn’s formula to perform better on rougher terrain.
    trigonometric: bool, optional, default: False
        whether to return trigonometric angle instead of azimuth. 
        Thus 0deg means East, 90deg North, 180deg West, 270deg South.

    Returns
    -------
    dst_raster: Raster
        Aspect calculated from the DEM.
    """
    options = dict(band=band+1, alg=alg,  trigonometric=trigonometric, format='MEM')
    ds_src = src_raster.to_gdal_ds()
    ds = gdal.DEMProcessing('', ds_src, 'aspect', **options)
    dst_raster = tgp.read_gdal_ds(ds)
    return dst_raster

def dem_to_TRI(src_raster, band=0, alg='Horn'):
    """Calculate the terrain ruggedness index (TRI) for the DEM.

    Parameters
    ----------
    src_raster : Raster
        The dem used to calculate the TRI.
    band : int, optional, default: 0
        source band number to use.
    alg : {'ZevenbergenThorne' or 'Horn'}, optional, default: Horn
        The literature suggests Zevenbergen & Thorne to be more suited to smooth landscapes, 
        where Horn’s formula to perform better on rougher terrain.

    Returns
    -------
    dst_raster: Raster
        TRI calculated from the DEM.
    """
    options = dict(band=band+1, alg=alg, format='MEM')
    ds_src = src_raster.to_gdal_ds()
    ds = gdal.DEMProcessing('', ds_src, 'TRI', **options)
    dst_raster = tgp.read_gdal_ds(ds)
    return dst_raster

def dem_to_TPI(src_raster, band=0, alg='Horn'):
    """Calculate the topographic position index (TPI) for the DEM.

    Parameters
    ----------
    src_raster : Raster
        The dem used to calculate the TPI.
    band : int, optional, default: 0
        source band number to use.
    alg : {'ZevenbergenThorne' or 'Horn'}, optional, default: Horn
        The literature suggests Zevenbergen & Thorne to be more suited to smooth landscapes, 
        where Horn’s formula to perform better on rougher terrain.

    Returns
    -------
    dst_raster: Raster
        TPI calculated from the DEM.
    """
    options = dict(band=band+1, alg=alg, format='MEM')
    ds_src = src_raster.to_gdal_ds()
    ds = gdal.DEMProcessing('', ds_src, 'TPI', **options)
    dst_raster = tgp.read_gdal_ds(ds)
    return dst_raster

def dem_to_roughness(src_raster, band=0, alg='Horn'):
    """Calculate the roughness for the DEM.

    Parameters
    ----------
    src_raster : Raster
        The dem used to calculate the roughness.
    band : int, optional, default: 0
        source band number to use.
    alg : {'ZevenbergenThorne' or 'Horn'}, optional, default: Horn
        The literature suggests Zevenbergen & Thorne to be more suited to smooth landscapes, 
        where Horn’s formula to perform better on rougher terrain.

    Returns
    -------
    dst_raster: Raster
        roughness calculated from the DEM.
    """
    options = dict(band=band+1, alg=alg, format='MEM')
    ds_src = src_raster.to_gdal_ds()
    ds = gdal.DEMProcessing('', ds_src, 'Roughness', **options)
    dst_raster = tgp.read_gdal_ds(ds)
    return dst_raster

#TODO
# 1. Plan Curvature, Profile Curvature
# 2. LS-Factor
#==================================
