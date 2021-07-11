FROM jeremy4555/gdal:3.0.4
MAINTAINER GoatWang

RUN apt-get install -y libspatialindex-dev
RUN pip3 install --upgrade pip

RUN pip3 install GDAL==3.0.4 Fiona==1.8.13 Shapely==1.6.4.post2 geopandas==0.7.0 Rtree==0.9.4 matplotlib==3.3.4 scipy==1.5.4 llvmlite
RUN pip3 install TronGisPy

