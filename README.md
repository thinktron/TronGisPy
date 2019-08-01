# Preinstlall
1. [gdal](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal): [Downlaod](https://download.lfd.uci.edu/pythonlibs/t4jqbe6o/GDAL-3.0.0-cp36-cp36m-win_amd64.whl)
1. [shapely](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely): [Downlaod](https://download.lfd.uci.edu/pythonlibs/t4jqbe6o/Shapely-1.6.4.post2-cp36-cp36m-win_amd64.whl)
1. [fiona](https://www.lfd.uci.edu/~gohlke/pythonlibs/#fiona):  [Downlaod](https://download.lfd.uci.edu/pythonlibs/t4jqbe6o/Fiona-1.8.6-cp36-cp36m-win_amd64.whl)
1. [Rtree](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rtree): [Downlaod](https://download.lfd.uci.edu/pythonlibs/t4jqbe6o/Rtree-0.8.3-cp36-cp36m-win_amd64.whl)
1. [geopandas](https://www.lfd.uci.edu/~gohlke/pythonlibs/#geopandas): [Downlaod](https://download.lfd.uci.edu/pythonlibs/t4jqbe6o/geopandas-0.5.0-py2.py3-none-any.whl)


# Install from pip server
```
pip install -U --index-url http://192.168.0.167:28181/simple --trusted-host 192.168.0.167 SplittedImage
```

# Build & deployee
```
python setup.py sdist bdist_wheel
scp dist\SplittedImage-0.0.3-py3-none-any.whl thinktron@rd.thinktronltd.com:/home/thinktron/pypi/SplittedImage-0.0.3-py3-none-any.whl
```


