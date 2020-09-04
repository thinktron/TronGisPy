import os
fps = ["docs\\source\\req_generator.rst", 
        "docs\\source\\setup.rst", 
        "docs\\source\\test.rst", 
        "docs\\source\\TronGisPy.rst", 
        "docs\\source\\TronGisPy.AeroTriangulation.rst", 
        "docs\\source\\TronGisPy.Algorithm.rst", 
        "docs\\source\\TronGisPy.CRS.rst", 
        "docs\\source\\TronGisPy.DEMProcessor.rst", 
        "docs\\source\\TronGisPy.GisIO.rst", 
        "docs\\source\\TronGisPy.Interpolation.rst", 
        "docs\\source\\TronGisPy.Normalizer.rst", 
        "docs\\source\\TronGisPy.Raster.rst", 
        "docs\\source\\TronGisPy.ShapeGrid.rst", 
        "docs\\source\\TronGisPy.SplittedImage.rst", 
        "docs\\source\\TronGisPy.TypeCast.rst", 
        "docs\\source\\TronGisPy.io.rst", 
        "docs\\source\\modules.rst"]
for f in  fps:
    if os.path.exists(f):
        os.remove(f)