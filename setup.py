import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TronGisPy",
    version="1.0.8",
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
          'matplotlib',
          'Rtree==0.9.4',
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
# 0.4.7 add resample alg in refine_resolution
# 0.4.8 fill in nan when reading data in get_nparray of GisIO
# 0.5.0 add ShapeGrid.py to process data in memory, get_nparray => GisIO.get_nparray in test.py
# 0.5.1 add get_extent in ShapeGrid.py 
# 0.5.2 clip_tif_by_bound => clip_tif_by_extent, remove projection fron clip_tif_by_extent params in ShapeGrid.py, return clipped geo_transform in clip_tif_by_extent & clip_tif_by_shp
# 0.5.3 remove projection from refine_resolution params in ShapeGrid.py
# 0.5.4 add geo_transform as return of refine_resolution function in ShapeGrid.py
# 0.5.5 adjust extent input of refine_resolution function in ShapeGrid.py
# 0.5.6 add AeroTriangulation.py and add fast majority interpolation in Interpolation.py
# 0.5.7 add fast mean interpolation in Interpolation.py
# 0.5.8 fast mean interpolation can be float type
# 0.5.9 add no_data_value in ShapeGrid.rasterize_layer
# 0.6.0 activate no_data_value in ShapeGrid.rasterize_layer
# 0.6.1 add loop_limit param in majority_interpolation & mean_interpolation functions 
# 0.6.2 modify geotransform setting in get_geo_attribute function in SplittedImage


# 1.0.0 Raster class for saving raster data in memory
# 1.0.1 update io, Raster, Interpolation, CRS, ShapeGrid documnetation
# 1.0.2 add astype function in Raster.
# 1.0.3 add get_values_by_coords function in Raster.
# 1.0.4 add np.bool in TypeCast, `gdal.FillNodata` in Interpolation
# 1.0.5 add data.copy() when assign data to Raster, and call update_gdaltype_by_npdtype() after assignment
# 1.0.6 add figsize when assign data to Raster, fill no_data when plotting Raster, add note in Interpolation.gdal_fillnodata
# 1.0.7 add gdaltype property in Raster, call paddedimage in `SplittedImage.apply` for efficiency.
# 1.0.8 add change gdaltype to gdaldtype, gdaltype property to gdaldtype_name in Raster, fill_na and fill_no_data in Raster, add fill_na param in `tgp.read_raster`


# python setup.py sdist bdist_wheel
# scp C:\Users\Thinktron\Projects\TronGisPy\dist\TronGisPy-1.0.8-py3-none-any.whl  jeremy@rd.thinktronltd.com:/home/ttl/pypi/TronGisPy-1.0.8-py3-none-any.whl
# scp C:\Users\TTL_R041\Desktop\Projects\RS2001\LineDetection\TronGisPy\dist\TronGisPy-1.0.8-py3-none-any.whl jeremy@rd.thinktronltd.com:/home/ttl/pypi/TronGisPy-1.0.8-py3-none-any.whl