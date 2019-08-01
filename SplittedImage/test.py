import unittest
import shutil

# data
import pandas as pd

# basic
import os
import numpy as np
from collections import Counter

# gis
from shapely.geometry import Polygon
import gdal

# main
from SplittedImage import SplittedImage

data_dir = os.path.join('data')
satellite_tif_dir = data_dir
satellite_tif_path = os.path.join(satellite_tif_dir, 'P0015913_SP5_006_001_002_021_002_005.tif')



class TestSplittedImage(unittest.TestCase):
    def setUp(self):
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        # window_size_h = window_size_w = step_size_h = step_size_w = 256
        self.box_size = 128
        ds = gdal.Open(satellite_tif_path)
        self.geo_transform = ds.GetGeoTransform()
        self.projection = ds.GetProjection()
        self.dtype_gdal = ds.GetRasterBand(1).DataType # gdal.GetDataTypeName(self.dtype_gdal)
        img_src_arr = ds.ReadAsArray()
        ds = None

        self.X = np.transpose(img_src_arr, axes=[1,2,0])
        self.splitted_image = SplittedImage(self.X, self.box_size, self.geo_transform, self.projection)

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def test___getitem__(self):
        slice_test1 = Counter(pd.cut(self.splitted_image[1].flatten(), bins=3, labels=range(3))) == Counter({1: 137343, 0: 122742, 2: 2059})
        slice_test2 = Counter(pd.cut(self.splitted_image[:2].flatten(), bins=3, labels=range(3))) == Counter({1: 412579, 0: 366496, 2: 7357})
        slice_test3 = Counter(pd.cut(self.splitted_image[:2, 2].flatten(), bins=3, labels=range(3))) == Counter({0: 97945, 1: 95721, 2: 2942})
        slice_test4 = Counter(pd.cut(self.splitted_image[:2, :2].flatten(), bins=3, labels=range(3))) == Counter({1: 333569, 0: 250291, 2: 5964})

        self.assertTrue(slice_test1)
        self.assertTrue(slice_test2)
        self.assertTrue(slice_test3)
        self.assertTrue(slice_test4)

    def test_get_padded_image(self):
        shape_test = self.splitted_image.padded_image.shape == (512, 512, 4)
        self.assertTrue(shape_test)

    def test_get_splitted_images(self):
        shape_test = self.splitted_image.get_splitted_images().shape == (16, 128, 128, 4)
        self.assertTrue(shape_test)

    def test_get_geo_attribute(self):
        df_attribute = self.splitted_image.get_geo_attribute()
        pol = df_attribute.loc[0, 'geometry']
        area_test = pol.area == 1638400.0
        self.assertTrue(area_test)

    def test_write_splitted_images(self):
        self.splitted_image.write_splitted_images(self.output_dir, 'P0015913_SP5_006_001_002_021_002_005')

    def test_write_combined_tif(self):
        box_size = 101
        splitted_image = SplittedImage(self.X, box_size, self.geo_transform, self.projection)
        X_pred = splitted_image.get_splitted_images()
        dst_tif_path = os.path.join(self.output_dir, "combined.tif")
        splitted_image.write_combined_tif(X_pred, dst_tif_path, self.dtype_gdal)


if __name__ == "__main__":
    unittest.main()
#  python -m unittest -v test.py