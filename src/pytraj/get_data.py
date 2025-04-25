#%% 
import rioxarray
import xarray as xr
import pandas as pd
import os
import glob
import psutil 
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from matplotlib import colors
import dask.array as da 

traj_list_all = ['No Data',
            'Loss without Alternation',
            'Gain without Alternation',
            'Loss with Alternation',
            'Gain with Alternation',
            'All Alternation Loss First',
            'All Alternation Gain First',
            'Stable Presence',
            'Stable Absence']



def get_data(years, 
             pres_val, 
             nodata_val,
             filepath, 
             areaunit = 'pixel',
             weight = False,
             chunk_size = 5000,
             type_ = 'raster',
             res = None):
    if(type_ == 'raster' or type_ == 'smallraster'):
        # if workpath indicates a tif file
        if (os.path.isfile(filepath) and filepath.lower().endswith('.tif')):
            if(nodata_val is not None):
                raster_data = rioxarray.open_rasterio(filepath).chunk({"band": -1, "y": chunk_size, "x": chunk_size})
            if(nodata_val is None or nodata_val is np.nan):
                raster_data = rioxarray.open_rasterio(filepath,masked=True).chunk({"band": -1, "y": chunk_size, "x": chunk_size})
        # if workpath indicates a folder 
        elif(os.path.isdir(filepath)):
            tif_files = sorted(glob.glob(f"{filepath}/**/*.tif", recursive=True))
            if(len(tif_files) != len(years)):
                raise ValueError("Number of files does not match number of time points!")  
            datasets = []
            for y,year in enumerate(years):
                # update filenname if needed 
                tif = tif_files[y]
                map_ = tif
                if(nodata_val is not None):
                    raster_map = rioxarray.open_rasterio(map_).chunk({"band": -1, "y": chunk_size, "x": chunk_size})
                if(nodata_val is None or nodata_val is np.nan):
                    raster_map = rioxarray.open_rasterio(map_,masked=True).chunk({"band": -1, "y": chunk_size, "x": chunk_size})
                raster_map = raster_map.assign_coords(band=[year])
                #raster_map.attrs["long_name"] = f"classification_{year}"
                datasets.append(raster_map)

            raster_data = xr.concat(datasets, dim='band')
            del datasets

        nt = raster_data.sizes['band']
        nr = raster_data.sizes['y']
        nc = raster_data.sizes['x']
        time_dim = raster_data.dims[0]
        res = raster_data.rio.resolution()
        params = {
            'pres_val':pres_val,
            'nodata_val':nodata_val,
            'years':years,
            'nt':nt,
            'nr':nr,
            'nc':nc,
            'time_dim':time_dim,
            'res':res,
            'areaunit':areaunit
        }

        return raster_data,params
    if(type_ == 'table'):
        input_ = pd.read_csv(filepath)
        input_ar = input_.iloc[:,1:-1].to_numpy().T
        ns = np.shape(input_)[0]
        nt = len(years)

        if(weight is True):
            weight = input_.iloc[:,-1].to_numpy()
        
        params = {
            'pres_val':pres_val,
            'nodata_val':nodata_val,
            'years':years,
            'nt':nt,
            'ns':ns,
            'weight':weight,
            'areaunit':areaunit,
            'res':res
        }

        return input_ar, params 