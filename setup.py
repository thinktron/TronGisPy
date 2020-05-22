import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TronGisPy",
    version="0.4.6",
    author="GoatWang",
    author_email="jeremywang@thinktronltd.com",
    description="Gis image Processing tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://rd.thinktronltd.com/jeremywang/TronGisPy",
    packages=setuptools.find_packages(),
    # package_data={'PySaga': ['saga_cmd_pkls/*']},
    # package_data={'PySatellite': ['data/*', 'data/*/*']},
    package_data={'TronGisPy': ['data/*', 'data/*/*']},
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
      install_requires=[
          'numba',
          'affine',
          'scikit-learn',
          'descartes',
          'matplotlib'
      ]
)

# 0.0.0
# 0.0.2 modify write_combined_tif to fit the image not well splitted by the right box size
# 0.0.3 add Normalizer
# 0.0.4 change name add CRS and SatelliteIO module
# 0.0.5 add kmeans in Algorithm, clip_image_by_shp & tif_composition in SatelliteIO
# 0.0.6 return no_data_value in get_geo_info
# 0.0.7 add refine_resolution in SatelliteIO
# 0.0.8 add polygonize & rasterize in SatelliteIO
# 0.0.9 add raster_pixel_to_polygon in SatelliteIO, transfer_npidx_to_coord, transfer_coord_to_npidx, transfer_npidx_to_coord_polygon in CRS
# 0.1.0 add get diff testing fp in SateloliteIO
# 0.1.1 add multipolygon in polygonize & zonal function
# 0.1.2 change zonal to default gdal polygonize function
# 0.1.3 add clip_shp_by_shp
# 0.1.4 add LineString in clip_shp_by_shp
# 0.1.5 write output tif can be without projection & geo_transform
# 0.1.6 get_extend return nparray not list

# 0.2.0 change name to TronGisPy
# 0.2.1 no need to give rows, cols, bend in write tif
# 0.2.2 add update_projection
# 0.2.3 cast npdtype to gdaldtype in TypeCast
# 0.2.4 set gdaldtyp in rasterize_layer function
# 0.2.5 change projection, gdadtype and no_data_value to write function in SplittedImage
# 0.2.6 fix SplittedImage.get_geo_attribute coord xy
# 0.2.7 add filter function in SplittedImage.write_splitted_image
# 0.2.8 add self.pad_val in SplittedImage init fun
# 0.2.9 add get_combined_image in SplittedImage
# 0.3.0 step size different from window size is acceptable in SplittedImage 
# 0.3.1 add reproject function in CRS.py
# 0.3.2 get_WKT_from epsg
# 0.3.3 add interpolation function
# 0.3.4 add clipper in Normalizer
# 0.3.5 get_epsg_from_wkt in CRS & remap_tif and reproject in GisIO
# 0.3.6 numba for coords sys transforming between npidx and coords
# 0.3.7 get_extend => get_extent, add param return_poly
# 0.3.8 numba_transfer_group_coord_to_npidx change return type 
# 0.3.9 add all_touched option in rasterize_layer
# 0.4.0 change interpolation function allowed shape to multiple bands image.
# 0.4.1 add transfer_npidx_to_coord
# 0.4.2 add DEMProcessor.py & crop_dem.tif as testing data
# 0.4.3 add function not in CRS
# 0.4.4 change CRS interface: numba_transfer_group_coord_to_npidx => transfer_group_coord_to_npidx
# 0.4.5 remove numba funcation & point to poly
# 0.4.6 transfer_npidx_to_coord_polygon using parallel algorithm

# python setup.py sdist bdist_wheel
# scp C:\Users\Thinktron\Projects\TronGisPy\dist\TronGisPy-0.4.6-py3-none-any.whl  thinktron@rd.thinktronltd.com:/home/thinktron/pypi/TronGisPy-0.4.6-py3-none-any.whl
# scp C:\Users\TTL_R041\Desktop\Projects\RS2001\LineDetection\TronGisPy\dist\TronGisPy-0.4.6-py3-none-any.whl thinktron@rd.thinktronltd.com:/home/thinktron/pypi/TronGisPy-0.4.6-py3-none-any.whl
