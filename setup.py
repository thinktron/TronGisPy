import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PySatellite",
    version="0.0.5",
    author="GoatWang",
    author_email="jeremywang@thinktronltd.com",
    description="Satellite image Processing tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://rd.thinktronltd.com/jeremywang/PySatellite",
    packages=setuptools.find_packages(),
    # package_data={'PySaga': ['saga_cmd_pkls/*']},
    package_data={'PySatellite': ['data/*']},
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
      install_requires=[
          'affine',
          'scikit-learn'
      ]
#    install_requires=[
#        "GDAL == 2.4.1",
#        "Shapely == 1.6.4.post2",
#        "Fiona == 1.8.6",
#        "Rtree >= 0.8.3",
#        "pyproj >= 2.2.2",
#        "geopandas >= 0.5.0",
#    ],
)

# 0.0.0
# 0.0.2 modify write_combined_tif to fit the image not well splitted by the right box size
# 0.0.3 add Normalizer
# 0.0.4 change name add CRS and SatelliteIO module
# 0.0.5 add kmeans in Algorithm, clip_image_by_shp & tif_composition in SatelliteIO

# pip3 install -U --index-url http://192.168.0.167:28181/simple --trusted-host 192.168.0.167 PySatellite
