import unittest

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
satellite_tif_path = os.path.join(satellite_tif_dir, 'P0015913_SP5_006_001_002.tif')

class TestSplittedImage(unittest.TestCase):
    def setUp(self):
        # window_size_h = window_size_w = step_size_h = step_size_w = 256
        box_size = 512
        ds = gdal.Open(satellite_tif_path)
        geo_transform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        img_src_arr = ds.ReadAsArray()
        ds = None

        X = np.transpose(img_src_arr, axes=[1,2,0])
        self.splitted_image = SplittedImage(X, box_size, geo_transform, projection)

    # def test___getitem__(self):
    #     slice_test1 = Counter(pd.cut(self.splitted_image[1].flatten(), bins=3, labels=range(3))) == Counter({0: 2373445, 1: 1818830, 2: 2029})
    #     slice_test2 = Counter(pd.cut(self.splitted_image[:2].flatten(), bins=3, labels=range(3))) == Counter({0: 6859244, 1: 5674081, 2: 49587})
    #     slice_test3 = Counter(pd.cut(self.splitted_image[:2, 2].flatten(), bins=3, labels=range(3))) == Counter({1: 395645, 0: 385327, 2: 5460})
    #     slice_test4 = Counter(pd.cut(self.splitted_image[:2, :2].flatten(), bins=3, labels=range(3))) == Counter({0: 1364836, 1: 993154, 2: 1306})

    #     self.assertTrue(slice_test1)
    #     self.assertTrue(slice_test2)
    #     self.assertTrue(slice_test3)
    #     self.assertTrue(slice_test4)

    # def test_get_padded_image(self):
    #     shape_test = self.splitted_image.padded_image.shape == (4096, 4096, 4)
    #     self.assertTrue(shape_test)

    # def test_get_splitted_images(self):
    #     shape_test = self.splitted_image.get_splitted_images().shape == (256, 256, 256, 4)
    #     self.assertTrue(shape_test)

    # def test_get_geo_attribute(self):
    #     df_attribute = self.splitted_image.get_geo_attribute()
    #     pol = df_attribute.loc[0, 'geometry']
    #     area_test = pol.area == 6553600.0
    #     self.assertTrue(area_test)

    def test_write_splitted_images(self):
        self.splitted_image.write_splitted_images('data', 'P0015913_SP5_006_001_002')




if __name__ == "__main__":
    unittest.main()
#  python -m unittest -v test.py