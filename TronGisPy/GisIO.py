import os
import numpy as np
from osgeo import ogr
from osgeo import osr
from osgeo import gdal
import TronGisPy as tgp
import geopandas as gpd
from collections import Counter
from TronGisPy import ShapeGrid
from shapely.geometry import Point, MultiPolygon, LineString, MultiLineString, Polygon

def clip_tif_by_shp(src_tif_path, src_shp_path, dst_tif_path):
    result = gdal.Warp(dst_tif_path,
                       src_tif_path,
                       cutlineDSName=src_shp_path,
                       cropToCutline=True)
    result = None

def clip_shp_by_shp(src_shp_path, clipper_shp_path, dst_shp_path):
    df_src = gpd.read_file(src_shp_path)
    df_clipper = gpd.read_file(clipper_shp_path)
    
    assert len(set([g.geom_type for g in  df_src['geometry']])) == 1, "geometry in the src_shp should have the same geom_type" 
    assert (len(set([g.geom_type for g in  df_clipper['geometry']])) == 1) and (df_clipper['geometry'].iloc[0].geom_type == 'Polygon'), "geom_type in the clipper_shp be Polygon" 

    geom_type = df_src['geometry'].iloc[0].geom_type
    if geom_type in ['Point', 'MultiPoint']:
        df_dst_shp = gpd.sjoin(df_src, df_clipper, how='inner')
    elif geom_type in ['Polygon', 'MultiPolygon']:
        df_dst_shp = gpd.overlay(df_src, df_clipper, how='intersection')
    elif geom_type in ['LineString', 'MultiLineString']:
        # TODO
        ## ogr solution: https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html#get-geometry-from-each-feature-in-a-layer
        ## geopandas solution: MultiLineString intersection will in trounble
        # poly_to_be_clipped_path = get_testing_fp('poly_to_be_clipped')
        # point_to_be_clipped_path = get_testing_fp('point_to_be_clipped')
        # shp_clipper_path = get_testing_fp('shp_clipper')
        # df_poly = gpd.read_file(poly_to_be_clipped_path)
        # df_point = gpd.read_file(point_to_be_clipped_path)
        # lines = [np.stack([np.ones((4))*i, np.arange(1, 5)]).T for i in [1,3,7]] + \
        #         [np.stack([np.ones((2))*i, np.arange(3, 5)]).T for i in [0.5,2,6]]
        # df_line = gpd.GeoDataFrame(geometry=[LineString(line) for line in lines])
        # df_multiline = gpd.GeoDataFrame(geometry=[MultiLineString(lines)])
        # df_clipper = gpd.read_file(shp_clipper_path)
        # 
        # fig, ax = plt.subplots(1,1)
        # df_clipper.plot(ax=ax)
        # df_line.plot(ax=ax, color='red')
        # 
        # df_line.intersection(df_clipper) 
        ## 1. num of clipper polygons will have different return rows
        ## 2. not intersect line will return None or "GEOMETRYCOLLECTION EMPTY"
        # df_multiline.intersection(df_clipper) 
        ## some part of WultiLine willdisappear
        # assert False, "We temporarily does not support for line strings clipping"
        multi_lines = []
        for line in df_src['geometry']:
            lines = []
            for poly in df_clipper['geometry']:
                line_intersection = line.intersection(poly)
                if not line_intersection.is_empty:
                    if line_intersection.geom_type == 'MultiLineString':
                        lines.extend(line.intersection(poly))
                    elif line_intersection.geom_type == 'LineString':
                        lines.append(line.intersection(poly))
            multi_lines.append(MultiLineString(lines))
        df_dst_shp = df_src.copy()
        df_dst_shp['geometry'] = multi_lines
        df_dst_shp.dropna(inplace=True)
    else:
        assert False, "geom_type must be Point, MultiPoint, Polygon, MultiPolygon, LineString or MultiLineString!"
    
    df_dst_shp.to_file(dst_shp_path)

def tif_composition(ref_tif_path, src_tif_paths, dst_tif_path, dst_tif_dtype_gdal=None):
    """
    ref_tif_path: should be used to create the canvas with final coordinate system, geo_transform and projection, 
    src_tif_paths: should be in list type with elements with full path of tif images.
    dst_tif_path: output file path
    """
    # get geo info
    rows, cols, bands, geo_transform, projection, dtype_gdal, no_data_value, metadata = tgp.get_raster_info(ref_tif_path)
    if dst_tif_dtype_gdal:
        dtype_gdal = dst_tif_dtype_gdal
        
    # cal bands count
    bands_for_each_tif = [tgp.get_raster_info(tif_path)[2] for tif_path in src_tif_paths]
    bands = sum(bands_for_each_tif)

    # bands compositions: create new tif
    dst_ds = gdal.GetDriverByName('GTiff').Create(dst_tif_path, cols, rows, bands, dtype_gdal)
    dst_ds.SetGeoTransform(geo_transform)
    dst_ds.SetProjection(projection)
    
    # bands compositions: write bands
    band_num = 1
    for tif_path, bands_for_the_tif in zip(src_tif_paths, bands_for_each_tif):
        nparr = tgp.get_raster_data(tif_path)
        for band_num_for_the_tif in range(bands_for_the_tif):
            band = dst_ds.GetRasterBand(band_num)
            band.WriteArray(nparr[:, :, band_num_for_the_tif], 0, 0)
            band.FlushCache()
            if no_data_value:
                band.SetNoDataValue(no_data_value)
            band_num += 1
    dst_ds = None

def refine_resolution(src_tif_path, dst_tif_path, dst_resolution, resample_alg='near'):
    """
    near: nearest neighbour resampling (default, fastest algorithm, worst interpolation quality).
    bilinear: bilinear resampling.
    cubic: cubic resampling.
    cubicspline: cubic spline resampling.
    lanczos: Lanczos windowed sinc resampling.
    average: average resampling, computes the weighted average of all non-NODATA contributing pixels.
    mode: mode resampling, selects the value which appears most often of all the sampled points.
    """
    result = gdal.Warp(dst_tif_path, src_tif_path, xRes=dst_resolution, yRes=dst_resolution, resampleAlg=resample_alg)
    result = None

def rasterize_layer(src_shp_path, dst_tif_path, ref_tif_path, use_attribute=None, all_touched=False, gdaldtype=None, no_data_value=None):
    """
    src_shp_path: rasterize which shp.
    dst_tif_path: rasterize output, should be in tiff type.
    ref_tif_path: the geo information reference raster.
    use_attribute: use thich attribute of the shp as raster value.
    """
    # Open your shapefile
    gdf_shp = gpd.read_file(src_shp_path)
    if not use_attribute:
        use_attribute = 'positive'
        gdf_shp[use_attribute] = 1


    # Create the destination raster data source
    # pixelWidth = pixelHeight = 2 # depending how fine you want your raster
    # x_min, x_max, y_min, y_max = source_layer.GetExtent()
    # cols = int((x_max - x_min) / pixelHeight)
    # rows = int((y_max - y_min) / pixelWidth)
    # geoTransform = (x_min, pixelWidth, 0, y_min, 0, pixelHeight)
    ref_tif_ds = gdal.Open(ref_tif_path)
    ref_tif_cols, ref_tif_rows = ref_tif_ds.RasterXSize, ref_tif_ds.RasterYSize
    ref_tif_geo_transofrm = ref_tif_ds.GetGeoTransform()

    rasterize_raster = ShapeGrid.rasterize_layer(gdf_shp, ref_tif_rows, ref_tif_cols, ref_tif_geo_transofrm, use_attribute, all_touched, no_data_value=no_data_value)
    rasterize_raster.to_file(dst_tif_path)


def vectorize_layer(src_tif_path, dst_shp_path, field_name='value', band_num=0, multipolygon=False):
    """band_num start from 1"""
    src_ds = gdal.Open(src_tif_path)
    srcband = src_ds.GetRasterBand(band_num+1)
    src_srs=osr.SpatialReference(wkt=src_ds.GetProjection())        
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(dst_shp_path)
    layer_name = os.path.split(dst_shp_path)[-1].replace(".shp", "")
    dst_layer = dst_ds.CreateLayer(layer_name, srs=src_srs)
    dst_layer.CreateField(ogr.FieldDefn(field_name, ogr.OFTInteger))
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex(field_name)
    gdal.Polygonize(srcband, None, dst_layer, dst_field, [], callback=None)

    src_ds = None
    dst_ds = None

    if multipolygon:
        df_shp = gpd.read_file(dst_shp_path)
        multi_polygons = df_shp.groupby(field_name)['geometry'].apply(list).apply(MultiPolygon)
        values = df_shp.groupby(field_name)[field_name].first()
        df_shp = gpd.GeoDataFrame(geometry=multi_polygons)
        df_shp[field_name] = values
        df_shp.to_file(dst_shp_path, index=False)

def raster_pixel_to_polygon(src_tif_path, dst_shp_path, all_bands_as_feature=False, crs=None, return_gdf=False):
    """
    crs should be dict type 'epsg:<epsg_code>', e.g. 'epsg:4326'
    """
    rows, cols, bands, geo_transform, projection, dtype_gdal, no_data_value, metadata = tgp.get_raster_info(src_tif_path)
    X = tgp.get_raster_data(src_tif_path)
    idxs = np.where(np.ones_like(X[:,:,0], dtype=bool))
    rows = []

    for row_idx, col_idx in zip(*idxs):
        row = {}
        npidx = (row_idx, col_idx)
        row['geometry'] = Polygon(tgp.npidxs_to_coord_polygons([npidx], geo_transform)[0])
        if all_bands_as_feature:
            for i in range(X.shape[2]):
                row['band'+str(i+1)] = X[row_idx, col_idx, i]
        rows.append(row)
    df_shp = gpd.GeoDataFrame(rows, geometry='geometry')
    if crs:
        df_shp.crs = crs
    if return_gdf:
       return df_shp
    else:
        df_shp.to_file(dst_shp_path)


def reproject(src_tif_path, dst_tif_path, dst_crs='EPSG:4326', src_crs=None):
    if src_crs:
        gdal.Warp(dst_tif_path, src_tif_path, srcSRS=src_crs, dstSRS=dst_crs)
    else:
        gdal.Warp(dst_tif_path, src_tif_path, dstSRS=dst_crs)


def remap_tif(src_tif_path, dst_tif_path, ref_tif_path):
    src_projection, src_gdaldtype = tgp.get_raster_info(src_tif_path, ['projection', 'gdaldtype'])
    ref_geo_transform, ref_projection = tgp.get_raster_info(ref_tif_path, ['geo_transform', 'projection'])
    output_bounds = tgp.get_raster_extent(ref_tif_path, 'gdal') # (minX, minY, maxX, maxY)

    x_res, y_res = ref_geo_transform[1], ref_geo_transform[5]
    output_type = src_gdaldtype
    gdal.Warp(dst_tif_path, src_tif_path, outputBounds=output_bounds,
                                            xRes=x_res,
                                            yRes=y_res,
                                            outputType=output_type,
                                            srcSRS=src_projection,
                                            dstSRS=ref_projection)



#TODO
# 1. raster pixel to points
