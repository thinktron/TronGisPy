# Preinstlall
## Install from thinktron pypi server
```
pip install -U --index-url http://192.168.0.167:28181/simple --trusted-host 192.168.0.167 Fiona
pip install -U --index-url http://192.168.0.167:28181/simple --trusted-host 192.168.0.167 GDAL
pip install -U --index-url http://192.168.0.167:28181/simple --trusted-host 192.168.0.167 Rtree
pip install -U --index-url http://192.168.0.167:28181/simple --trusted-host 192.168.0.167 Shapely
pip install -U --index-url http://192.168.0.167:28181/simple --trusted-host 192.168.0.167 pyproj
pip install -U --index-url http://192.168.0.167:28181/simple --trusted-host 192.168.0.167 geopandas
```

## Install from wheel
1. [gdal](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal): [Downlaod](https://download.lfd.uci.edu/pythonlibs/t4jqbe6o/GDAL-2.4.1-cp36-cp36m-win_amd64.whl)
1. [shapely](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely): [Downlaod](https://download.lfd.uci.edu/pythonlibs/t4jqbe6o/Shapely-1.6.4.post2-cp36-cp36m-win_amd64.whl)
1. [fiona](https://www.lfd.uci.edu/~gohlke/pythonlibs/#fiona):  [Downlaod](https://download.lfd.uci.edu/pythonlibs/t4jqbe6o/Fiona-1.8.6-cp36-cp36m-win_amd64.whl)
1. [Rtree](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rtree): [Downlaod](https://download.lfd.uci.edu/pythonlibs/t4jqbe6o/Rtree-0.8.3-cp36-cp36m-win_amd64.whl)
1. [geopandas](https://www.lfd.uci.edu/~gohlke/pythonlibs/#geopandas): [Downlaod](https://download.lfd.uci.edu/pythonlibs/t4jqbe6o/geopandas-0.5.0-py2.py3-none-any.whl)

# Install from pip server
```bash
pip install -U --index-url http://192.168.0.167:28181/simple --trusted-host 192.168.0.167 SplittedImage
```

# Build & deployee
```bash
python setup.py sdist bdist_wheel
scp C:\Users\Thinktron\Projects\SplittedImage\dist\SplittedImage-0.0.2-py3-none-any.whl  thinktron@rd.thinktronltd.com:/home/thinktron/pypi/SplittedImage-0.0.2-py3-none-any.whl
```

# Usage
```python
import os
from PySatellite.SplittedImage import SplittedImage
from PySatellite.SatelliteIO import get_geo_info, get_nparray, get_extend, get_testing_fp
from PySatellite.CRS import transfer_xy_to_coord, transfer_coord_to_xy
satellite_tif_path = get_testing_fp()

box_size = 128 # split into (128, 128) shape images
cols, rows, bands, geo_transform, projection, dtype_gdal = get_geo_info(satellite_tif_path)
X = get_nparray(satellite_tif_path)
splitted_image = SplittedImage(X, box_size, geo_transform, projection)

# in order to make the image divisible, image will be padded
padded_image = splitted_image.padded_image 
print(padded_image.shape)

# get splitted image as numpy array
splitted_images = splitted_image.get_splitted_images()
print(splitted_images.shape) #(num_splitted_images, box_size, box_size, num_bands)

# get the basic information (idx, idx_h, idx_w, geo_transform, geometry) of each splitted images
output_dir = "output"
if not os.path.isdir(output_dir): os.mkdir(output_dir)
df_attribute = splitted_image.get_geo_attribute()
df_attribute.to_file(os.path.join(output_dir, "file_dst.shp"))

# write the splitted images in to disk
splitted_image.write_splitted_images(output_dir, 'P0015913_SP5')

# once you successfully run AI model on each image, you can combine it into one
X_pred = splitted_images
dst_tif_path = os.path.join(output_dir, "result_combined.tif")
splitted_image.write_combined_tif(X_pred, dst_tif_path, dtype_gdal)
```

