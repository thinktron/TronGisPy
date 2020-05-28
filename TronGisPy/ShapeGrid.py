import ogr
import gdal
import numpy as np
import geopandas as gpd
from TronGisPy import TypeCast

def read_gdal_ds(ds, numpy_shape=True, fill_na=False):
    """if numpy_shape the shape will be (cols, rows, bnads), else (bnads, cols, rows)"""
    X = ds.ReadAsArray()
    ds = None 
    if fill_na:
        X = X.astype(np.float)
        no_data_value = get_geo_info(fp)[6]
        assert no_data_value is not None, "no_data_value is None which cannot be filled!"
        X[X == no_data_value] = np.nan

    if not numpy_shape:
        return X
    else:
        if len(X.shape) == 2:
            X = X.reshape(-1, *X.shape)
        return np.transpose(X, axes=[1,2,0])

def build_gdal_ds(X=None, bands=None, cols=None, rows=None, geo_transform=None, projection=None, gdaldtype=None, no_data_value=None):
    """X should be in (n_rows, n_cols, n_bands) shape"""
    if X is None:
        assert (bands is not None) and (cols is not None) and (rows is not None), "bands, cols, rows should not be None"
        assert (gdaldtype is not None), "gdaldtype should not be None"
    else:    
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=2)
        bands = X.shape[2] if bands is None else bands
        cols = X.shape[1] if cols is None else cols
        rows = X.shape[0] if rows is None else rows
        gdaldtype = TypeCast.convert_npdtype_to_gdaldtype(X.dtype) if gdaldtype is None else gdaldtype

    ds = gdal.GetDriverByName('MEM').Create('', cols, rows, bands, gdaldtype) # dst_filename, xsize=512, ysize=512, bands=1, eType=gdal.GDT_Byte
    if geo_transform is not None:
        ds.SetGeoTransform(geo_transform)
    if projection is not None:
        ds.SetProjection(projection)

    if X is not None:
        for b in range(X.shape[2]):
            band = ds.GetRasterBand(b+1)
            band.WriteArray(X[:, :, b], 0, 0)
            if no_data_value:
                band.SetNoDataValue(no_data_value)
            band.FlushCache()

    return ds

def rasterize_layer(gdf_shp, rows, cols, geo_ransform, use_attribute, all_touched=False):
    """
    gdf_shp: should be GeoDataFrame type
    rows, cols, geo_ransform: output raster geo_info
    use_attribute: use this attribute of the shp as raster value.
    all_touched: pixels that touch (not overlap over 50%) the poly will be the value of the poly.
    """
    # Open your shapefile
    assert type(gdf_shp) is gpd.GeoDataFrame, "gdf_shp should be GeoDataFrame type."
    assert use_attribute in gdf_shp.columns, "attribute not exists in gdf_shp."
    gdaldtype = TypeCast.convert_npdtype_to_gdaldtype(gdf_shp[use_attribute].dtype)
    src_shp_ds = ogr.Open(gdf_shp.to_json())
    src_shp_layer = src_shp_ds.GetLayer()

    # Create the destination raster data source
    dst_ds = build_gdal_ds(bands=1, cols=cols, rows=rows, geo_transform=geo_ransform, gdaldtype=gdaldtype)

    # set it to the attribute that contains the relevant unique
    options = ["ATTRIBUTE="+use_attribute]
    if all_touched:
        options.append('ALL_TOUCHED=TRUE')
    gdal.RasterizeLayer(dst_ds, [1], src_shp_layer, options=options) # target_ds, band_list, source_layer, options = options

    out_arr = dst_ds.GetRasterBand(1).ReadAsArray()
    return out_arr

def clip_tif_by_shp(X, src_shp_path, geo_transform, projection):
    src_ds = build_gdal_ds(X, geo_transform=geo_transform, projection=projection)
    dst_ds = gdal.Warp('',
                       src_ds,
                       format= 'MEM',
                       cutlineDSName=src_shp_path,
                       cropToCutline=True)
    arr = read_gdal_ds(dst_ds)
    return arr

def clip_tif_by_bounds(X, bounds, geo_transform, projection):
    """
    outputBounds --- output bounds as (minX, minY, maxX, maxY) in target SRS
    """
    src_ds = build_gdal_ds(X, geo_transform=geo_transform, projection=projection)
    dst_ds = gdal.Warp('',
                       src_ds,
                       format= 'MEM',
                       outputBounds=bounds,
                       cropToCutline=True)
    arr = read_gdal_ds(dst_ds)
    return arr


def refine_resolution(X, geo_transform, projection, dst_resolution, resample_alg='near'):
    """
    near: nearest neighbour resampling (default, fastest algorithm, worst interpolation quality).
    bilinear: bilinear resampling.
    cubic: cubic resampling.
    cubicspline: cubic spline resampling.
    lanczos: Lanczos windowed sinc resampling.
    average: average resampling, computes the weighted average of all non-NODATA contributing pixels.
    mode: mode resampling, selects the value which appears most often of all the sampled points.
    """
    src_ds = build_gdal_ds(X, geo_transform=geo_transform, projection=projection)
    dst_ds = gdal.Warp('', src_ds, xRes=dst_resolution, yRes=dst_resolution, format='MEM', resampleAlg=resample_alg)
    arr = read_gdal_ds(dst_ds)
    return arr