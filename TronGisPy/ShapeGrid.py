import os
import ogr
import gdal
import pyproj
import shapely
import numpy as np
import geopandas as gpd
import TronGisPy as tgp

def rasterize_layer(src_vector, rows, cols, geo_transform, use_attribute, all_touched=False, no_data_value=0):
    """
    src_vector: should be GeoDataFrame type
    rows, cols, geo_transform: output raster geo_info
    use_attribute: use this attribute of the shp as raster value.
    all_touched: pixels that touch (not overlap over 50%) the poly will be the value of the poly.
    return Raster
    """
    # Open your shapefile
    assert type(src_vector) is gpd.GeoDataFrame, "src_vector should be GeoDataFrame type."
    assert use_attribute in src_vector.columns, "attribute not exists in src_vector."
    gdaldtype = tgp.npdtype_to_gdaldtype(src_vector[use_attribute].dtype)
    # projection = src_vector.crs.to_wkt() if src_vector.crs is not None else None
    projection = pyproj.CRS(src_vector.crs).to_wkt() if src_vector.crs is not None else None
    src_shp_ds = ogr.Open(src_vector.to_json())
    src_shp_layer = src_shp_ds.GetLayer()

    # Create the destination raster data source
    ds = tgp.write_gdal_ds(bands=1, cols=cols, rows=rows, geo_transform=geo_transform, gdaldtype=gdaldtype, no_data_value=no_data_value)

    # set it to the attribute that contains the relevant unique
    options = ["ATTRIBUTE="+use_attribute]
    if all_touched:
        options.append('ALL_TOUCHED=TRUE')
    gdal.RasterizeLayer(ds, [1], src_shp_layer, options=options) # target_ds, band_list, source_layer, options = options

    data = ds.GetRasterBand(1).ReadAsArray()
    raster = tgp.Raster(data, geo_transform, projection, gdaldtype, no_data_value)
    return raster

def vectorize_layer(src_raster, field_name='value', band_num=1, multipolygon=False):
    """band_num start from 1"""
    src_ds = src_raster.to_gdal_ds()
    src_band = src_ds.GetRasterBand(band_num)

    drv = ogr.GetDriverByName("MEMORY")
    dst_ds = drv.CreateDataSource('memData')
    dst_layer = dst_ds.CreateLayer('vectorize_layer')
    dst_layer.CreateField(ogr.FieldDefn(field_name, ogr.OFTInteger))
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex(field_name)
    gdal.Polygonize(src_band, None, dst_layer, dst_field, [], callback=None)

    df_vectorized_rows = []
    for feature in dst_layer:
        field_value = feature.GetField(field_name)
        geometry = shapely.wkt.loads(feature.GetGeometryRef().ExportToWkt())
        df_vectorized_rows.append([field_value, geometry])
    df_vectorized = gpd.GeoDataFrame(df_vectorized_rows, columns=[field_name, 'geometry'], geometry='geometry', crs=src_raster.projection)
    src_ds = None
    dst_ds = None

    if multipolygon:
        from shapely.geometry import MultiPolygon
        multi_polygons = df_vectorized.groupby(field_name)['geometry'].apply(list).apply(MultiPolygon)
        values = df_vectorized.groupby(field_name)[field_name].first()
        df_vectorized = gpd.GeoDataFrame(geometry=multi_polygons)
        df_vectorized[field_name] = values

    return df_vectorized


def clip_raster_with_polygon(src_raster, src_shp):
    """
    src_shp: should be GeoDataFrame type
    """
    assert src_raster.geo_transform is not None, "src_raster.geo_transform should not be None"
    src_ds = src_raster.to_gdal_ds()
    temp_dir = tgp.create_temp_dir()
    src_shp_fp = os.path.join(temp_dir, 'src_shp.shp')
    src_shp.to_file(src_shp_fp)
    dst_ds = gdal.Warp('', src_ds, format= 'MEM', cutlineDSName=src_shp_fp, cropToCutline=True)
    dst_raster = tgp.read_gdal_ds(dst_ds)
    tgp.remove_temp_dir()
    return dst_raster

def clip_raster_with_extent(src_raster, extent):
    """
    extent --- output bounds as (minX, minY, maxX, maxY) in target SRS
    """
    assert src_raster.geo_transform is not None, "src_raster.geo_transform should not be None"
    src_ds = src_raster.to_gdal_ds()
    dst_ds = gdal.Warp('', src_ds, format= 'MEM', outputBounds=extent, cropToCutline=True)
    dst_raster = tgp.read_gdal_ds(dst_ds)
    return dst_raster

def refine_resolution(src_raster, dst_resolution, resample_alg='near', extent=None):
    """
    near: nearest neighbour resampling (default, fastest algorithm, worst interpolation quality).
    bilinear: bilinear resampling.
    cubic: cubic resampling.
    cubicspline: cubic spline resampling.
    lanczos: Lanczos windowed sinc resampling.
    average: average resampling, computes the weighted average of all non-NODATA contributing pixels.
    mode: mode resampling, selects the value which appears most often of all the sampled points.
    """
    src_ds = src_raster.to_gdal_ds()
    dst_ds = gdal.Warp('', src_ds, xRes=dst_resolution, yRes=dst_resolution, outputBounds=extent, format='MEM', resampleAlg=resample_alg)
    dst_raster = tgp.read_gdal_ds(dst_ds)
    return dst_raster