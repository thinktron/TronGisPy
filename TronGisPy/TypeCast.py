import gdal
import numpy as np
import pandas as pd

df_gdal_dtype = [
    [0, "GDT_Unknown", None],
    [1, "GDT_Byte", np.int8],
    [2, "GDT_UInt16", np.uint16],
    [3, "GDT_Int16", np.int16],
    [4, "GDT_UInt32", np.uint32],
    [5, "GDT_Int32", np.int32],
    [6, "GDT_Float32", np.float32],
    [7, "GDT_Float64", np.float64],
    [8, "GDT_CInt16", None],
    [9, "GDT_CInt32", None],
    [10, "GDT_CFloat32", None],
    [11, "GDT_CFloat64", None],
    [12, "GDT_TypeCount", None],
]
df_gdal_dtype = pd.DataFrame(df_gdal_dtype, columns=['gdaldtype_idx', 'gdaldtype_name', 'npdtype'])

def get_gdaldtype_name(gdaldtype_idx):
    gdaldtype_name = df_gdal_dtype.loc[df_gdal_dtype['gdaldtype_idx']==gdaldtype_idx, 'gdaldtype_name'].iloc[0]
    return gdaldtype_name

def gdaldtype_to_npdtype(gdaldtype_idx):
    npdtype = df_gdal_dtype.loc[df_gdal_dtype['gdaldtype_idx']==gdaldtype_idx, 'npdtype'].iloc[0]
    return npdtype

def npdtype_to_gdaldtype(npdtype): 
    if npdtype in df_gdal_dtype['npdtype'].tolist():
        return int(df_gdal_dtype.loc[df_gdal_dtype['npdtype']==npdtype, 'gdaldtype_idx'].iloc[0])
    elif np.issubdtype(npdtype, np.signedinteger):
        return gdal.GDT_Int16
    elif np.issubdtype(npdtype, np.unsignedinteger):
        return gdal.GDT_UInt16
    elif np.issubdtype(npdtype, np.floating):
        return gdal.GDT_Float32
