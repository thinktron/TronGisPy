import os

write_str = """"""
base_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(base_dir, "requirements.txt"), 'r' , encoding="utf8") as f:
    for line in f:
        if not ("Fiona" in line or "GDAL" in line or "Shapely" in line or "geopandas" in line or "Rtree" in line):
            write_str += line

write_str += "--index-url http://rd.thinktronltd.com:28181/simple --trusted-host rd.thinktronltd.com\n"
write_str += "GDAL==3.0.4\n"
write_str += "Fiona==1.8.13\n"
write_str += "Shapely==1.6.4.post2 \n"
write_str += "geopandas==0.7.0 \n"
write_str += "Rtree==0.9.4\n"

# pip install Fiona-1.8.6-cp36-cp36m-win_amd64.whl Rtree-0.8.3-cp36-cp36m-win_amd64.whl Shapely-1.6.4.post1-cp36-cp36m-win_amd64.whl GDAL-2.4.1-cp36-cp36m-win_amd64.whl geopandas-0.5.0-py2.py3-none-any.whl
with open("requirements_new.txt", 'w' , encoding="utf8") as f:
    f.write(write_str)