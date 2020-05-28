import unittest

# basic
import os
import time
import shutil
import numpy as np

# data
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# gis
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import gdal

# main
from TronGisPy.SplittedImage import SplittedImage
from TronGisPy.Normalizer import Normalizer
from TronGisPy import GisIO, Algorithm, CRS, TypeCast, Interpolation, DEMProcessor, ShapeGrid

data_dir = os.path.join('TronGisPy', 'data')
satellite_tif_path = GisIO.get_testing_fp('satellite_tif')
satellite_tif_clipper_path = GisIO.get_testing_fp('satellite_tif_clipper')
satellite_tif_kmeans_path = GisIO.get_testing_fp('satellite_tif_kmeans')
rasterized_image_path = GisIO.get_testing_fp('rasterized_image')
rasterized_image_1_path = GisIO.get_testing_fp('rasterized_image_1')
poly_to_be_clipped_path = GisIO.get_testing_fp('poly_to_be_clipped')
point_to_be_clipped_path = GisIO.get_testing_fp('point_to_be_clipped')
line_to_be_clipped_path = GisIO.get_testing_fp('line_to_be_clipped')
multiline_to_be_clipped_path = GisIO.get_testing_fp('multiline_to_be_clipped')
remap_rgb_clipper_path = GisIO.get_testing_fp('remap_rgb_clipper_path')
remap_ndvi_path = GisIO.get_testing_fp('remap_ndvi_path')

shp_clipper_path = GisIO.get_testing_fp('shp_clipper')
dem_process_path = GisIO.get_testing_fp('dem_process_path')
# interpolation_points_path = os.path.join(data_dir, 'interpolation', 'climate_points.shp')

# show_image = True
show_image = False

class TestSplittedImage(unittest.TestCase):
    def setUp(self):
        time.sleep(1)
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        # window_size_h = window_size_w = step_size_h = step_size_w = 256
        # self.box_size = 128
        # self.step_size = 64
        self.box_size = 254
        self.step_size = 127
        
        cols, rows, bands, geo_transform, projection, gdaldtype, no_data_value = GisIO.get_geo_info(satellite_tif_path)
        self.geo_transform = geo_transform
        self.projection = projection
        self.gdaldtype = gdaldtype
        self.no_data_value = no_data_value
        self.X = GisIO.get_nparray(satellite_tif_path)

        self.splitted_image = SplittedImage(self.X, self.box_size, self.geo_transform, step_size=self.step_size)

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        time.sleep(1)

    def test___getitem__(self):
        self.assertTrue(self.splitted_image.convert_to_inner_index_h(0,0) == (0, 254))
        self.assertTrue(self.splitted_image.convert_to_inner_index_h(1,1) == (127, 381))
        self.assertTrue(self.splitted_image.convert_to_inner_index_h(2,2) == (254, 508))
        self.assertTrue(Counter(pd.cut(self.splitted_image[1].flatten(), bins=3, labels=range(3))) == Counter({0: 268790, 1: 247772, 2: 3630}))
        self.assertTrue(Counter(pd.cut(self.splitted_image[:2].flatten(), bins=3, labels=range(3))) == Counter({1: 523687, 0: 508289, 2: 8408}))
        self.assertTrue(Counter(pd.cut(self.splitted_image[:2, 2].flatten(), bins=3, labels=range(3))) == Counter({0: 285697, 1: 225614, 2: 4817}))
        self.assertTrue(Counter(pd.cut(self.splitted_image[:2, :2].flatten(), bins=3, labels=range(3))) == Counter({1: 521447, 0: 502404, 2: 8405}))

    def test_get_padded_image(self):
        shape_test = self.splitted_image.padded_image.shape == (635, 635, 4)
        self.assertTrue(shape_test)

    def test_get_splitted_images(self):
        shape_test = self.splitted_image.get_splitted_images().shape == (16, 254, 254, 4)
        self.assertTrue(shape_test)

    def test_get_geo_attribute(self):
        df_attribute = self.splitted_image.get_geo_attribute()
        df_attribute.to_file(os.path.join(self.output_dir, "df_attribute.shp"))
        pol = df_attribute.loc[0, 'geometry']
        area_test = pol.area == 6451600.0
        self.assertTrue(area_test)

    def test_write_splitted_images(self):
        self.splitted_image.write_splitted_images(self.output_dir, 'test_satellite', projection=self.projection, gdaldtype=self.gdaldtype, no_data_value=self.no_data_value)

    def test_get_combined_image(self):
        X_pred = self.splitted_image.get_splitted_images()
        X_combined = self.splitted_image.get_combined_image(X_pred, padding=3, aggregator='mean')

    def test_write_combined_tif(self):
        X_pred = self.splitted_image.get_splitted_images()
        dst_tif_path = os.path.join(self.output_dir, "combined.tif")
        self.splitted_image.write_combined_tif(X_pred, dst_tif_path, projection=self.projection, gdaldtype=self.gdaldtype)

class TestCRS(unittest.TestCase):
    def setUp(self):
        time.sleep(1)
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        time.sleep(1)

    def test_transfer_npidx_to_coord(self):
        cols, rows, bands, geo_transform, projection, gdaldtype, no_data_value = GisIO.get_geo_info(satellite_tif_path)
        npidx = (1,3)
        coord = CRS.transfer_npidx_to_coord(npidx, geo_transform)
        self.assertTrue(coord == (328560.0, 2750780.0))

    def test_transfer_coord_to_npidx(self):
        cols, rows, bands, geo_transform, projection, gdaldtype, no_data_value = GisIO.get_geo_info(satellite_tif_path)
        coord = (328560.0+9, 2750780.0-9) # resolution is 10 meter, add 9 will be in the same cell
        npidx = CRS.transfer_coord_to_npidx(coord, geo_transform)
        self.assertTrue(npidx == (1, 3))
        coord = (328560.0+11, 2750780.0-11) # resolution is 10 meter, add 11 will be in the next cell
        npidx = CRS.transfer_coord_to_npidx(coord, geo_transform)
        self.assertTrue(npidx == (2, 4))

    def test_transfer_npidx_to_coord_polygon(self):
        cols, rows, bancds, geo_transform, projection, gdaldtype, no_data_value = GisIO.get_geo_info(satellite_tif_path)
        npidx = [0,2]
        polygon = CRS.transfer_npidx_to_coord_polygon(npidx, geo_transform)
        # df_lands_boundry = gpd.GeoDataFrame([{'geometry':polygon}], geometry='geometry', crs={'init' :'epsg:3826'})
        # dst_shp_path = os.path.join(self.output_dir, 'df_lands_boundry.shp')
        # df_lands_boundry.to_file(dst_shp_path)
        centroid = polygon.centroid.x, polygon.centroid.y
        self.assertTrue(centroid == (328555.0, 2750785.0))

    def test_get_wkt_from_epsg(self):
        WKT4326 = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
        target_WKT = CRS.get_wkt_from_epsg(4326)
        self.assertTrue(WKT4326 == target_WKT)

    def test_transfer_group_coord_to_npidx(self):
        cols, rows, bands, geo_transform, projection, gdaldtype, no_data_value = GisIO.get_geo_info(satellite_tif_path)
        coords = np.array([(328560.0+9, 2750780.0-9)]) # resolution is 10 meter, add 9 will be in the same cell
        npidx = CRS.transfer_group_coord_to_npidx(coords, geo_transform)
        self.assertTrue(np.sum(npidx == np.array([(1, 3)])) == 2)
        coords = np.array([(328560.0+11, 2750780.0-11)]) # resolution is 10 meter, add 11 will be in the next cell
        npidx = CRS.transfer_group_coord_to_npidx(coords, geo_transform)
        self.assertTrue(np.sum(npidx == np.array([(2, 4)])) == 2)

    def test_transfer_group_npidx_to_coord(self):
        cols, rows, bands, geo_transform, projection, gdaldtype, no_data_value = GisIO.get_geo_info(satellite_tif_path)
        npidxs = [(1,3)]
        coords = CRS.transfer_group_npidx_to_coord(npidxs, geo_transform)
        self.assertTrue(np.sum(coords == np.array([(328560.0, 2750780.0)])) == 2)

class TestGisIO(unittest.TestCase):
    def setUp(self):
        time.sleep(1)
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        time.sleep(1)

    def test_get_nparray(self):
        dst_image_path = os.path.join(self.output_dir, 'clipped_image.tif')
        GisIO.clip_tif_by_shp(satellite_tif_path, satellite_tif_clipper_path, dst_image_path)
        clip_image_arr = GisIO.get_nparray(dst_image_path, fill_na=True)
        self.assertTrue(np.sum(np.isnan(clip_image_arr)) == 42144)

    def test_get_extent(self):
        cols, rows, bands, geo_transform, projection, gdaldtype, no_data_value = GisIO.get_geo_info(satellite_tif_path)
        extent = GisIO.get_extent(satellite_tif_path, False)
        self.assertTrue(extent == (328530.0, 333650.0, 2745670.0, 2750790.0))

    def test_clip_tif_by_shp(self):
        dst_image_path = os.path.join(self.output_dir, 'clipped_image.tif')
        GisIO.clip_tif_by_shp(satellite_tif_path, satellite_tif_clipper_path, dst_image_path)
        clip_image_arr = GisIO.get_nparray(dst_image_path)
        if show_image:
            plt.imshow(clip_image_arr)
            plt.title("TestGisIO" + ": " + "test_clip_tif_by_shp")
            plt.show()
        self.assertTrue(clip_image_arr.shape == (138, 225, 4))

    def test_clip_shp_by_shp(self):
        # polygon clipping
        src_shp_path = poly_to_be_clipped_path
        clipper_shp_path = shp_clipper_path
        dst_shp_path = os.path.join(self.output_dir, 'clipped_poly.shp')
        GisIO.clip_shp_by_shp(src_shp_path, clipper_shp_path, dst_shp_path)
        gdf_clipped_poly = gpd.read_file(dst_shp_path)
        self.assertTrue(gdf_clipped_poly['geometry'].apply(lambda x:x.area).sum() == 3)

        # points clipping
        src_shp_path = point_to_be_clipped_path
        clipper_shp_path = shp_clipper_path
        dst_shp_path = os.path.join(self.output_dir, 'clipped_point.shp')
        GisIO.clip_shp_by_shp(src_shp_path, clipper_shp_path, dst_shp_path)
        gdf_clipped_point = gpd.read_file(dst_shp_path)
        self.assertTrue(gdf_clipped_point['geometry'].iloc[0].coords[0] == (3,3))

        # line clipping
        src_shp_path = line_to_be_clipped_path
        clipper_shp_path = shp_clipper_path
        dst_shp_path = os.path.join(self.output_dir, 'clipped_line.shp')
        GisIO.clip_shp_by_shp(src_shp_path, clipper_shp_path, dst_shp_path)
        gdf_clipped_line = gpd.read_file(dst_shp_path)
        self.assertTrue(np.sum([line.length for line in gdf_clipped_line['geometry'] if line != None]) == 4)

        # multiline clipping
        src_shp_path = multiline_to_be_clipped_path
        clipper_shp_path = shp_clipper_path
        dst_shp_path = os.path.join(self.output_dir, 'clipped_multiline.shp')
        GisIO.clip_shp_by_shp(src_shp_path, clipper_shp_path, dst_shp_path)
        gdf_clipped_multiline = gpd.read_file(dst_shp_path)
        self.assertTrue(np.sum([line.length for line in gdf_clipped_line['geometry'] if line != None]) == 4)

    def test_tif_composition(self):
        crs_tif_image = satellite_tif_path
        src_tif_paths = [satellite_tif_path, satellite_tif_kmeans_path]
        dst_tif_path = os.path.join(self.output_dir, 'composited_image.tif')
        GisIO.tif_composition(crs_tif_image, src_tif_paths, dst_tif_path)

        composited_image_arr = GisIO.get_nparray(dst_tif_path)
        if show_image:
            plt.imshow(composited_image_arr[:, :, 4], cmap='gray')
            plt.title("TestGisIO" + ": " + "test_tif_composition")
            plt.show()

        self.assertTrue(composited_image_arr.shape == (512, 512, 5))

    def test_refine_resolution(self):
        src_tif_path = satellite_tif_path
        dst_tif_path = os.path.join(self.output_dir, 'resolution_refined_image.tif')
        dst_resolution = 5
        GisIO.refine_resolution(src_tif_path, dst_tif_path, dst_resolution, 'bilinear')

        resolution_refined_image_arr = GisIO.get_nparray(dst_tif_path)
        if show_image:
            plt.imshow(resolution_refined_image_arr)
            plt.title("TestGisIO" + ": " + "test_refine_resolution")
            plt.show()
        self.assertTrue(resolution_refined_image_arr.shape == (1024, 1024, 4))

    def test_update_geo_info(self):
        dst_tif_path = os.path.join(self.output_dir, 'X_geo_info_updated.tif')
        shutil.copyfile(satellite_tif_path, dst_tif_path)
        cols, rows, bands, geo_transform, projection, gdaldtype, no_data_value = GisIO.get_geo_info(dst_tif_path)
        projection = CRS.get_wkt_from_epsg(3826)
        geo_transform = list(geo_transform)
        geo_transform[0] += 10
        geo_transform[3] -= 10
        GisIO.update_geo_info(dst_tif_path, projection=projection, geo_transform=geo_transform)
        cols, rows, bands, geo_transform, projection, gdaldtype, no_data_value = GisIO.get_geo_info(dst_tif_path)
        self.assertTrue(geo_transform == (328540.0, 10.0, 0.0, 2750780.0, 0.0, -10.0))
        self.assertTrue(projection == 'PROJCS["TWD97 / TM2 zone 121",GEOGCS["TWD97",DATUM["Taiwan_Datum_1997",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","1026"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","3824"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",121],PARAMETER["scale_factor",0.9999],PARAMETER["false_easting",250000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],AUTHORITY["EPSG","3826"]]')

    def test_write_output_tif(self):
        dst_image_path = os.path.join(self.output_dir, 'clipped_image.tif')
        GisIO.clip_tif_by_shp(satellite_tif_path, satellite_tif_clipper_path, dst_image_path)

        cols, rows, bands, geo_transform, projection, gdaldtype, no_data_value = GisIO.get_geo_info(dst_image_path)
        clip_image_arr = GisIO.get_nparray(dst_image_path)
        padded_image_arr = np.pad(clip_image_arr, ((0,62), (0,75), (0,0)), mode='constant', constant_values=0)
        dst_tif_path = os.path.join(self.output_dir, 'padded_image.tif')
        GisIO.write_output_tif(padded_image_arr,dst_tif_path,4,300,200,geo_transform, projection)
        padded_image_arr = GisIO.get_nparray(dst_tif_path)
        if show_image:
            plt.imshow(padded_image_arr)
            plt.title("TestGisIO" + ": " + "test_write_output_tif")
            plt.show()
        self.assertTrue(padded_image_arr.shape == (200, 300, 4))

        # test write output without projection & geotransform
        X = np.random.rand(10000).reshape(100,100)
        dst_tif_path = os.path.join(self.output_dir, "test_output.tif")
        GisIO.write_output_tif(X, dst_tif_path, 1, 100, 100, gdaldtype=gdal.GDT_Float32)
        test_output = GisIO.get_nparray(dst_tif_path)
        self.assertTrue(test_output.shape == (100, 100, 1))

    def test_rasterize_layer(self):
        src_shp_path = satellite_tif_clipper_path
        dst_tif_path = os.path.join(self.output_dir, 'rasterized_image.tif')
        ref_tif_path = satellite_tif_path
        GisIO.rasterize_layer(src_shp_path, dst_tif_path, ref_tif_path)
        rasterized_image = GisIO.get_nparray(dst_tif_path)
        if show_image:
            plt.imshow(rasterized_image[:,:,0], cmap='gray')
            plt.title("TestGisIO" + ": " + "test_rasterize_layer")
            plt.show()
        self.assertTrue(np.sum(rasterized_image==1) == 20512)

        GisIO.rasterize_layer(src_shp_path, dst_tif_path, ref_tif_path, all_touched=True)
        rasterized_image = GisIO.get_nparray(dst_tif_path)
        if show_image:
            plt.imshow(rasterized_image[:,:,0], cmap='gray')
            plt.title("TestGisIO" + ": " + "test_rasterize_layer")
            plt.show()
        self.assertTrue(np.sum(rasterized_image==1) == 20876)

    def test_polygonize_layer(self):
        src_tif_path = rasterized_image_path
        dst_shp_path = os.path.join(self.output_dir, 'polygonized_layer.shp')
        GisIO.polygonize_layer(src_tif_path, dst_shp_path)
        df_shp = gpd.read_file(dst_shp_path)
        if show_image:
            df_shp.plot()
            plt.show()
        self.assertTrue(df_shp.loc[0, 'geometry'].area == 2051200)

        src_tif_path = rasterized_image_1_path
        GisIO.polygonize_layer(src_tif_path, dst_shp_path, remove_boundry=False, multipolygon=True)
        df_shp = gpd.read_file(dst_shp_path)
        if show_image:
            df_shp.plot()
            plt.show()
        self.assertTrue(df_shp.loc[0, 'geometry'].area == 3624400.0)

    def test_raster_pixel_to_polygon(self):
        src_tif_path = satellite_tif_path
        dst_shp_path = os.path.join(self.output_dir, 'raster_pixel_to_polygon.shp')
        GisIO.raster_pixel_to_polygon(src_tif_path, dst_shp_path, all_bands_as_feature=True, crs={'init' :'epsg:3826'})

    def test_reproject(self):
        src_tif_path = satellite_tif_path
        dst_tif_path = os.path.join(self.output_dir, "X_reprojected.tif")
        GisIO.reproject(src_tif_path, dst_tif_path, dst_crs='EPSG:3826')
        WKT3826 = 'PROJCS["TWD97 / TM2 zone 121",GEOGCS["TWD97",DATUM["Taiwan_Datum_1997",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","1026"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","3824"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",121],PARAMETER["scale_factor",0.9999],PARAMETER["false_easting",250000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],AUTHORITY["EPSG","3826"]]'
        self.assertTrue(GisIO.get_geo_info(dst_tif_path)[4] == WKT3826)

        cols, rows, bands, geo_transform, projection, gdaldtype, no_data_value = GisIO.get_geo_info(dst_tif_path)
        tif_without_projection_path = os.path.join(self.output_dir, "X_without_projection.tif")
        GisIO.write_output_tif(GisIO.get_nparray(dst_tif_path), tif_without_projection_path, geo_transform=geo_transform, gdaldtype=gdaldtype)

        src_tif_path = tif_without_projection_path
        dst_tif_path = os.path.join(self.output_dir, "X_reprojected_2.tif")
        GisIO.reproject(src_tif_path, dst_tif_path, dst_crs='EPSG:4326', src_crs='EPSG:3826')
        WKT4326 = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]'
        self.assertTrue(GisIO.get_geo_info(dst_tif_path)[4] == WKT4326)
    
    def test_remap_tif(self):
        src_tif_path = GisIO.get_testing_fp('remap_ndvi_path')
        ref_tif_path = GisIO.get_testing_fp('remap_rgb_clipper_path')
        dst_tif_path = os.path.join(self.output_dir, 'X_remapped.tif')
        GisIO.remap_tif(src_tif_path, dst_tif_path, ref_tif_path)

        ref_cols, ref_rows, ref_bands, ref_geo_transform, ref_projection, ref_gdaldtype, ref_no_data_value = GisIO.get_geo_info(ref_tif_path)
        dst_cols, dst_rows, dst_bands, dst_geo_transform, dst_projection, dst_gdaldtype, dst_no_data_value = GisIO.get_geo_info(dst_tif_path)
        self.assertTrue((ref_cols, ref_rows) == (dst_cols, dst_rows))
        self.assertTrue(ref_geo_transform == dst_geo_transform)
        self.assertTrue(ref_projection == dst_projection)

    def test_get_testing_fp(self):
        fn = 'satellite_tif'
        fp = GisIO.get_testing_fp(fn)
        self.assertTrue(fp == 'C:\\Users\\Thinktron\\Projects\\TronGisPy\\TronGisPy\\data\\satellite_tif\\satellite_tif.tif')
        
        fn = 'satellite_tif_clipper'
        fp = GisIO.get_testing_fp(fn)
        self.assertTrue(fp == 'C:\\Users\\Thinktron\\Projects\\TronGisPy\\TronGisPy\\data\\satellite_tif_clipper\\satellite_tif_clipper.shp')

        fn = 'satellite_tif_kmeans'
        fp = GisIO.get_testing_fp(fn)
        self.assertTrue(fp == 'C:\\Users\\Thinktron\\Projects\\TronGisPy\\TronGisPy\\data\\satellite_tif_kmeans\\satellite_tif_kmeans.tif')

        fn = 'rasterized_image'
        fp = GisIO.get_testing_fp(fn)
        self.assertTrue(fp == 'C:\\Users\\Thinktron\\Projects\\TronGisPy\\TronGisPy\\data\\rasterized_image\\rasterized_image.tif')

class TestNormalizer(unittest.TestCase):
    def setUp(self):
        time.sleep(1)
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        self.cols, self.rows, self.bands, self.geo_transform, self.projection, self.gdaldtype, self.no_data_value = GisIO.get_geo_info(satellite_tif_path)
        self.X = GisIO.get_nparray(satellite_tif_path)

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        time.sleep(1)

    def test_Normalizer(self):
        X_norm = Normalizer().fit_transform(self.X) 
        self.assertTrue(np.sum(X_norm==1) == 289)
        self.assertTrue(np.sum(X_norm==0) == 1)

        X_norm = Normalizer().fit_transform(self.X, min_max_val=(20, 240)) 
        self.assertTrue(np.sum(X_norm==1) == 727)
        self.assertTrue(np.sum(X_norm==0) == 76876)

        X_norm = Normalizer().fit_transform(self.X, clip_percentage=0.1) 
        self.assertTrue(np.sum(X_norm==1) == 105926)
        self.assertTrue(np.sum(X_norm==0) == 112995)

class TestAlgorithm(unittest.TestCase):
    def setUp(self):
        time.sleep(1)
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        self.cols, self.rows, self.bands, self.geo_transform, self.projection, self.gdaldtype, self.no_data_value = GisIO.get_geo_info(satellite_tif_path)
        self.X = GisIO.get_nparray(satellite_tif_path)

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        time.sleep(1)

    def test_kmeans(self):
        X_kmeans = Algorithm.kmeans(self.X, n_clusters=5, no_data_value=0)
        dst_tif_path = os.path.join(self.output_dir, "X_kmeans.tif")
        bands = 1
        GisIO.write_output_tif(X_kmeans, dst_tif_path, bands, self.cols, self.rows, self.geo_transform, self.projection)

        kmeans_image_arr = GisIO.get_nparray(dst_tif_path)
        if show_image:
            plt.imshow(kmeans_image_arr[:, :, 0], cmap='gray')
            plt.title("TestAlgorithm" + ": " + "test_kmeans")
            plt.show()
        self.assertTrue(Counter(list(np.hstack(kmeans_image_arr[:, :, 0])))[4] == 9511)

class TestTypeCast(unittest.TestCase):
    def setUp(self):
        time.sleep(1)
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        time.sleep(1)

    def test_get_gdaldtype_name_by_idx(self):
        self.assertTrue(TypeCast.get_gdaldtype_name_by_idx(5) == 'GDT_Int32')

    def test_convert_gdaldtype_to_npdtype(self):
        self.assertTrue(TypeCast.convert_gdaldtype_to_npdtype(5) == np.int32)

    def test_convert_npdtype_to_gdaldtype(self):
        self.assertTrue(TypeCast.convert_npdtype_to_gdaldtype(np.int32) == 5)

class TestInterpolation(unittest.TestCase):
    def setUp(self):
        time.sleep(1)
        X = GisIO.get_nparray(satellite_tif_path).astype(np.float)
        raw_shape = X.shape
        X = X.flatten()
        rand_idx = np.random.randint(0, len(X), int(len(X)*0.3))
        X[rand_idx] = np.nan
        self.X = X.reshape(raw_shape)
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        time.sleep(1)

    def test_img_interpolation(self):
        X_band0 = self.X[:, :, 0]
        X_band0_interp_nearest = Interpolation.img_interpolation(X_band0, method='nearest')
        X_band0_interp_linear = Interpolation.img_interpolation(X_band0, method='linear')
        X_band0_interp_cubic = Interpolation.img_interpolation(X_band0, method='cubic')
        if show_image:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(X_band0, cmap='gray')
            axes[1].imshow(X_band0_interp_nearest, cmap='gray')
            axes[2].imshow(X_band0_interp_linear, cmap='gray')
            axes[3].imshow(X_band0_interp_cubic, cmap='gray')
            
        self.assertTrue(np.sum(np.isnan(X_band0_interp_nearest)) < np.product(X_band0_interp_nearest.shape) * 0.05)
        self.assertTrue(np.sum(np.isnan(X_band0_interp_linear)) < np.product(X_band0_interp_nearest.shape) * 0.05)
        self.assertTrue(np.sum(np.isnan(X_band0_interp_cubic)) < np.product(X_band0_interp_nearest.shape) * 0.05)
    
class TestDEMProcessor(unittest.TestCase):
    def setUp(self):
        time.sleep(1)
        self.dem_cols, self.dem_rows, self.dem_bands, self.dem_geo_transform, self.dem_projection, self.dem_gdaldtype, self.dem_no_data_value = GisIO.get_geo_info(dem_process_path)
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        time.sleep(1)

    def test_dem_to_hillshade(self):
        dst_tif_path = os.path.join(self.output_dir, 'hillshade.tif')
        DEMProcessor.dem_to_hillshade(dem_process_path, dst_tif_path, azimuth=0, altitude=30)
        dst_cols, dst_rows, dst_bands, dst_geo_transform, dst_projection, dst_gdaldtype, dst_no_data_value = GisIO.get_geo_info(dst_tif_path)
        dst_array = GisIO.get_nparray(dst_tif_path)

        self.assertTrue((self.dem_cols, self.dem_rows) == (dst_cols, dst_rows))
        self.assertTrue(self.dem_geo_transform == dst_geo_transform)
        self.assertTrue(self.dem_no_data_value == dst_no_data_value)
        self.assertGreaterEqual(np.nanmin(dst_array), 0)
        self.assertLessEqual(np.nanmax(dst_array), 255)

    def test_dem_to_slope(self):
        dst_tif_path = os.path.join(self.output_dir, 'slope.tif')
        DEMProcessor.dem_to_slope(dem_process_path, dst_tif_path, slope_format='degree')
        dst_cols, dst_rows, dst_bands, dst_geo_transform, dst_projection, dst_gdaldtype, dst_no_data_value = GisIO.get_geo_info(dst_tif_path)
        dst_array = GisIO.get_nparray(dst_tif_path)

        self.assertTrue((self.dem_cols, self.dem_rows) == (dst_cols, dst_rows))
        self.assertTrue(self.dem_geo_transform == dst_geo_transform)
        self.assertTrue(self.dem_no_data_value == dst_no_data_value)
        self.assertGreaterEqual(np.nanmin(dst_array), 0)
        self.assertLessEqual(np.nanmax(dst_array), 90)
    
    def test_dem_to_aspect(self):
        dst_tif_path = os.path.join(self.output_dir, 'aspect.tif')
        DEMProcessor.dem_to_aspect(dem_process_path, dst_tif_path)
        dst_cols, dst_rows, dst_bands, dst_geo_transform, dst_projection, dst_gdaldtype, dst_no_data_value = GisIO.get_geo_info(dst_tif_path)
        dst_array = GisIO.get_nparray(dst_tif_path)

        self.assertTrue((self.dem_cols, self.dem_rows) == (dst_cols, dst_rows))
        self.assertTrue(self.dem_geo_transform == dst_geo_transform)
        self.assertTrue(self.dem_no_data_value == dst_no_data_value)
        self.assertGreaterEqual(np.nanmin(dst_array), 0)
        self.assertLessEqual(np.nanmax(dst_array), 360)
    
    def test_dem_to_TRI(self):
        dst_tif_path = os.path.join(self.output_dir, 'tri.tif')
        DEMProcessor.dem_to_TRI(dem_process_path, dst_tif_path)
        dst_cols, dst_rows, dst_bands, dst_geo_transform, dst_projection, dst_gdaldtype, dst_no_data_value = GisIO.get_geo_info(dst_tif_path)
        dst_array = GisIO.get_nparray(dst_tif_path)

        self.assertTrue((self.dem_cols, self.dem_rows) == (dst_cols, dst_rows))
        self.assertTrue(self.dem_geo_transform == dst_geo_transform)
        self.assertTrue(self.dem_no_data_value == dst_no_data_value)
        self.assertGreaterEqual(np.nanmin(dst_array), 0)

    def test_dem_to_TPI(self):
        dst_tif_path = os.path.join(self.output_dir, 'tpi.tif')
        DEMProcessor.dem_to_TPI(dem_process_path, dst_tif_path)
        dst_cols, dst_rows, dst_bands, dst_geo_transform, dst_projection, dst_gdaldtype, dst_no_data_value = GisIO.get_geo_info(dst_tif_path)
        dst_array = GisIO.get_nparray(dst_tif_path)

        self.assertTrue((self.dem_cols, self.dem_rows) == (dst_cols, dst_rows))
        self.assertTrue(self.dem_geo_transform == dst_geo_transform)
        self.assertTrue(self.dem_no_data_value == dst_no_data_value)
    
    def test_dem_to_roughness(self):
        dst_tif_path = os.path.join(self.output_dir, 'roughness.tif')
        DEMProcessor.dem_to_roughness(dem_process_path, dst_tif_path)
        dst_cols, dst_rows, dst_bands, dst_geo_transform, dst_projection, dst_gdaldtype, dst_no_data_value = GisIO.get_geo_info(dst_tif_path)
        dst_array = GisIO.get_nparray(dst_tif_path)

        self.assertTrue((self.dem_cols, self.dem_rows) == (dst_cols, dst_rows))
        self.assertTrue(self.dem_geo_transform == dst_geo_transform)
        self.assertTrue(self.dem_no_data_value == dst_no_data_value)
        self.assertGreaterEqual(np.nanmin(dst_array), 0)

class TestShapeGrid(unittest.TestCase):
    def setUp(self):
        time.sleep(1)
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        time.sleep(1)

    def test_clip_tif_by_shp(self):
        X = GisIO.get_nparray(satellite_tif_path)
        cols, rows, bands, geo_transform, projection, gdaldtype, no_data_value = GisIO.get_geo_info(satellite_tif_path)
        X_clipped, geo_transform_clipped = ShapeGrid.clip_tif_by_shp(X, geo_transform, projection, satellite_tif_clipper_path)
        if show_image:
            plt.imshow(X_clipped)
            plt.title("TestShapeGrid" + ": " + "test_clip_tif_by_shp")
            plt.show()
        self.assertTrue(X_clipped.shape == (138, 225, 4))
        self.assertTrue(geo_transform_clipped == (329460.0, 10.0, 0.0, 2748190.0, 0.0, -10.0))
        
    def test_clip_tif_by_extent(self):
        X = GisIO.get_nparray(satellite_tif_path)
        cols, rows, bands, geo_transform, projection, gdaldtype, no_data_value = GisIO.get_geo_info(satellite_tif_path)
        gdf = gpd.read_file(satellite_tif_clipper_path)
        X_clipped, geo_transform_clipped = ShapeGrid.clip_tif_by_extent(X, geo_transform, gdf.total_bounds)
        if show_image:
            fig, (ax1 ,ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(X)
            ax2.imshow(X_clipped)
            plt.title("TestShapeGrid" + ": " + "test_clip_tif_by_bounds")
            plt.show()
        self.assertTrue(X_clipped.shape == (138, 226, 4))
        self.assertTrue(geo_transform_clipped == (329405.4857111082, 10.00517198347499, 0.0, 2748181.2743610824, 0.0, -10.010622679670949))

    def test_refine_resolution(self):
        X = GisIO.get_nparray(satellite_tif_path)
        cols, rows, bands, geo_transform, projection, gdaldtype, no_data_value = GisIO.get_geo_info(satellite_tif_path)
        X_refined = ShapeGrid.refine_resolution(X, geo_transform, projection, dst_resolution=5, resample_alg='bilinear')
        if show_image:
            plt.imshow(X_refined)
            plt.title("TestShapeGrid" + ": " + "test_refine_resolution")
            plt.show()
        self.assertTrue(X_refined.shape == (1024, 1024, 4))

    def test_get_extent(self):
        cols, rows, bands, geo_transform, projection, gdaldtype, no_data_value = GisIO.get_geo_info(satellite_tif_path)
        extent = ShapeGrid.get_extent(rows, cols, geo_transform, False)
        self.assertTrue(extent == (328530.0, 333650.0, 2745670.0, 2750790.0))

if __name__ == "__main__":
    unittest.main()
#  python -m unittest -v test.py