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
1. [gdal](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)
1. [shapely](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely)
1. [fiona](https://www.lfd.uci.edu/~gohlke/pythonlibs/#fiona)
1. [Rtree](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rtree)
1. [geopandas](https://www.lfd.uci.edu/~gohlke/pythonlibs/#geopandas)

# Install from pip server
```
pip install -U --index-url http://192.168.0.167:28181/simple --trusted-host 192.168.0.167 PySatellite
```

# Usage
Please see [Tutorial.ipynb](http://rd.thinktronltd.com:21111/jeremywang/PySatellite/blob/master/Tutorial.ipynb)

# Build & deployee
```bash
python setup.py sdist bdist_wheel
scp C:\Users\Thinktron\Projects\PySatellite\dist\PySatellite-0.1.0-py3-none-any.whl  thinktron@rd.thinktronltd.com:/home/thinktron/pypi/PySatellite-0.1.0-py3-none-any.whl
```


