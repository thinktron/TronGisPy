import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TronGisPy",
    version="0.2.0",
    author="GoatWang",
    author_email="jeremywang@thinktronltd.com",
    description="Gis image Processing tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://rd.thinktronltd.com/jeremywang/PySatellite",
    packages=setuptools.find_packages(),
    # package_data={'PySaga': ['saga_cmd_pkls/*']},
    package_data={'PySatellite': ['data/*', 'data/*/*']},
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
      install_requires=[
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

# 0.2.1 change name to TronGisPy

# python setup.py sdist bdist_wheel
# scp C:\Users\Thinktron\Projects\TronGisPy\dist\TronGisPy-0.2.0-py3-none-any.whl  thinktron@rd.thinktronltd.com:/home/thinktron/pypi/TronGisPy-0.2.0-py3-none-any.whl