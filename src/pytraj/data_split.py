#%%
from dask.distributed import Client, LocalCluster
cluster = LocalCluster(n_workers=4, threads_per_worker=1,processes = False)
client = Client(cluster)
client

# %%
def split_tiles(raster_data):
    # Get the number of rows (y dimension) and columns (x dimension)
    num_rows, num_cols = raster_data.sizes['y'], raster_data.sizes['x']
    
    # Define the size of each tile (5x5 grid)
    tile_height = num_rows // tile_row  # Rows per tile
    tile_width = num_cols // tile_col   # Columns per tile
    
    tile_idx = 0
    
    # Loop through the 5x5 grid of tiles
    for i in range(tile_row):
        for j in range(tile_col):
            print(i,j)
            # Calculate the index range for this tile
            row_start = i * tile_height
            row_end = (i + 1) * tile_height if i != tile_row - 1 else num_rows
            col_start = j * tile_width
            col_end = (j + 1) * tile_width if j != tile_col - 1 else num_cols
            
            # Slice the data for this tile based on indices
            tile_data = raster_data.isel(y=slice(row_start, row_end), x=slice(col_start, col_end))
            
            valid_data = tile_data != nodata_val 
            #print(tile_data)

            
            if(valid_data.any().compute()):
                tiled_data.append(tile_data) 
    
    return tile_data 


#%% read splitted data 
import dask.array as da
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import rioxarray 


pres_val = 3
nodata_val = 0
chunk_size = 1000
years = np.arange(1985,2022,1).astype(str)
data_dir = "D:/atlantic_forest/"
tif_files = glob.glob(f"{data_dir}/AF_tile_*.tif")

# Dictionary to store datasets with tile number as key
datasets = {}

for tif in tif_files:
    # Extract tile number from filename (assuming "AF_tile_xx.tif")
    tile_num = int(tif.split("_")[-1].split(".")[0])

    # Load with rioxarray and store in dictionary
    datasets[tile_num] = rioxarray.open_rasterio(tif).chunk({"band": -1, "y": chunk_size, "x": chunk_size})

chunking = (datasets[0].shape[0], chunk_size,chunk_size)
nt = datasets[0].sizes['band']
#%%
traj_list_all = ['No Data',
            'Loss without Alternation',
            'Gain without Alternation',
            'Loss with Alternation',
            'Gain with Alternation',
            'All Alternation Loss First',
            'All Alternation Gain First',
            'Stable Presence',
            'Stable Absence']


def reclass(input_,pres_val,nodata_val,chunking = None):
    reclassed = da.full_like(input_, fill_value=1, dtype="ubyte")

    # Apply conditions using Dask's `where()` (NumPy-compatible)
    reclassed = da.where(input_ == pres_val, 2, reclassed)  # Set 2 where combined_data == 3
    reclassed = da.where(input_ == nodata_val, 0, reclassed)  # Set 0 where combined_data == 0

    #Chunk the array properly
    #reclassed = raster_map.rechunk(chunking)

    return reclassed


def get_change(input_):
    
    if np.all(input_ == 0):  # Skip background chunks
        #print("all nan")
        return np.full_like(input_[1:], fill_value=-2, dtype="int8")
    else: 
        change_ar = np.diff(input_,axis = 0).astype("int8")
        #adjacent_sum = ref_ar[:,:-1] + ref_ar[:,1:]
        adjacent_sum = input_[:-1,:,:] + input_[1:,:,:]
        change_ar[adjacent_sum == 2*2] = 2 # stable presence 
        change_ar[adjacent_sum == 1*2] = 0 # stable absence 
        change_ar[input_[1:] == 0] = -2 # no data 
        return change_ar



def get_traj(change_block):
    

    if np.all(change_block == nodata_val):  # Skip background chunks
        #print("all nan")
        return np.zeros_like(change_block[0]).astype('int8')

    else: 

        # ref_change: 0 - stable absence, -1, loss, 1, gain, 2 stable presence
        nr = change_block.shape[1]
        nc = change_block.shape[2]
        # identify trajectories 
        traj = np.zeros((nr,nc),dtype = "int8")

        # count number of gains and losses 
        num_gain = (change_block == 1).sum(axis=0).astype("int8")
        num_loss = (change_block == -1).sum(axis=0).astype("int8")
        num_pre = (change_block == 2).sum(axis=0).astype("int8")
        num_abs = (change_block == 0).sum(axis=0).astype("int8")
        num_alt = np.minimum(num_gain, num_loss)  # Find the minimum between gain and loss

        first_loss = np.argmax(change_block == -1, axis=0).astype("int8")
        first_loss[num_loss == 0] = nt # give it an out of bound number
        first_gain = np.argmax(change_block == 1, axis = 0).astype("int8")
        first_gain[num_gain == 0] = nt
        

        # traj 1: loss without alternation (only one loss, no gain)
        traj[(num_gain == 0) & (num_loss == 1)] = 1
        # traj 2: gain without alternation (only one gain, no loss)
        traj[(num_gain == 1) & (num_loss == 0)] = 2
        # traj 3: loss with altneration (num loss > num alt)
        traj[(num_alt > 0) & (num_loss > num_alt)] = 3
        # traj 4: gain with alternation (num gain > num alt)
        traj[(num_alt > 0) & (num_gain > num_alt)] = 4
        # traj 5: all alternation loss first (num loss = num loss, first_loss < first_gain)
        traj[(num_alt > 0) & (num_gain == num_loss) & (first_loss < first_gain)] = 5
        # traj 6: all alternation gain first (num gain = num alt, first_gain < first_loss)
        traj[(num_alt > 0) & (num_gain == num_loss) & (first_gain < first_loss)] = 6
        # traj 7: stable presence (num_pre = nc-1)
        traj[num_pre == nt-1] = 7
        # traj 8: stable absence (num_abs = nc-1)
        traj[num_abs == nt-1] = 8
        traj[np.any(change_block == -2, axis=0)] = 0

        #xr_traj = xr.zeros_like(change_block[0]).astype('int8')
        #xr_traj.values = traj
        return traj
    #return xr.DataArray(traj, dims=('y', 'x'), coords={'y': raster_map.coords['y'], 'x': raster_map.coords['x']})


#%%
import xarray as xr
weight = False 
annual = True
areaunit = 'perc_region'
tiled = True
res = 30
traj_list = traj_list_all[1:-2]
traj_all = []

years = np.arange(1985,2022,1).astype(str)
nt = len(years)
traj_loss_all = pd.DataFrame(index=years[:-1], columns=traj_list)
traj_loss_all = traj_loss_all.fillna(0)
traj_gain_all = pd.DataFrame(index=years[:-1], columns=traj_list)
traj_gain_all = traj_gain_all.fillna(0)
global sum_rregion, sum_extent,quantity_all,exchange_all,alternation_all
sum_rregion = 0
sum_extent = 0
quantity_all,exchange_all,alternation_all = 0,0,0
from tqdm import tqdm
import time

for ti in tqdm(range(len(datasets)), desc="Processing Tiles"):

    start_time = time.time()  # Start timing
    print(f"Processing tile {ti}...")
    #print("processing tile", ti)
    tile_da = datasets[ti].data
    reclassed_ = reclass(tile_da,pres_val,nodata_val,chunking = chunking)

    # Define the template (must match expected output shape)
    change_template = da.zeros_like(reclassed_[1:], dtype="int8")
    # Apply Dask map_blocks (directly on the Dask array)
    ref_change = da.map_blocks(
        get_change, 
        reclassed_, 
        dtype="int8",
        meta = change_template,
        new_axis = None,
        chunks =(reclassed_.chunks[0][0] - 1,) + reclassed_.chunks[1:])
    
    xr_traj = xr.zeros_like(datasets[ti][0]).astype('int8')
    traj = da.map_blocks(get_traj,ref_change,dtype = "int8",drop_axis = 0)
    xr_traj.data = traj

    traj_all.append(xr_traj)

    # ------------------- Compute components ------------------- # 
    bichange = (reclassed_[-1] - reclassed_[0]).astype("int8")
    bigain = (bichange == 1).sum().compute()
    biloss = (bichange == -1).sum().compute()
    quantity = bigain - biloss
    exchange = min(bigain,biloss)

    quantity_all += quantity
    exchange_all += exchange

    totalgain = (ref_change == 1).sum()
    totalloss = (ref_change == -1).sum()
    totalchange = (totalgain + totalloss).compute()
    alternation = totalchange - quantity-exchange 
    alternation_all += alternation

    traj_loss_ = pd.DataFrame(index=years[:-1], columns=traj_list)
    traj_gain_ = pd.DataFrame(index=years[:-1], columns=traj_list)
    traj_loss = traj_loss_.copy()
    traj_gain = traj_gain_.copy()
    traj_loss_a = traj_loss_.copy()
    traj_gain_a = traj_gain_.copy()
    diff_years = np.diff(np.array(years).astype("int8"))

    if(areaunit == 'perc_region'):
        
        size_rregion = np.sum(np.any(reclassed_ == 2,axis = 0)).compute()
        sum_rregion = sum_rregion+size_rregion


    if(areaunit == 'perc_extent'):
        size_extent = np.sum(np.any(reclassed_ != 0,axis = 0)).compute()
        sum_extent = sum_extent + size_extent 

    for i,t in enumerate(traj_list):
        change_i = da.where(traj == (i + 1), ref_change, 0)

        if(weight is not False):
            weight_i = weight[traj == i+1]
            traj_loss_.iloc[:,i] = -1*np.sum((change_i == -1).astype(int) * weight_i,axis = 1)
            traj_gain_.iloc[:,i] = np.sum((change_i == 1).astype(int) * weight_i,axis = 1)
            
        elif(weight == False):
            print("processing traj",t)
            traj_loss_.iloc[:,i] = -1* np.sum(change_i == -1, axis = (1,2)).compute()
            traj_gain_.iloc[:,i] = np.sum(change_i == 1, axis = (1,2)).compute()
        
        if(areaunit == 'perc_region'):
            #print("computing percentage for traj",t)
            # compute relevant region 
            if(tiled is True):
                traj_loss = traj_loss_
                traj_gain = traj_gain_
            else:
                traj_loss.iloc[:,i] = 100*traj_loss_.iloc[:,i]/size_rregion
                traj_gain.iloc[:,i] = 100*traj_gain_.iloc[:,i]/size_rregion

        if(areaunit == 'perc_extent'):
            if(tiled is True):
                traj_loss = traj_loss_
                traj_gain = traj_gain_
            else:
                traj_loss.iloc[:,i] = 100*traj_loss_.iloc[:,i]/size_extent
                traj_gain.iloc[:,i] = 100*traj_gain_.iloc[:,i]/size_extent            
        elif(areaunit == 'km2'):
            #traj_loss_km = traj_loss_.copy()
            #traj_gain_km = traj_gain_.copy()
            traj_loss.iloc[:,i] = (traj_loss_[:,i]*(res[0]*res[1]))/(1000**2)
            traj_gain.iloc[:,i] = (traj_gain_[:,i]*(res[0]*res[1]))/(1000**2)
            #traj_loss = traj_loss_km
            #traj_gain = traj_gain_km
        elif(areaunit == 'pixels'):
            traj_loss = traj_loss_
            traj_gain = traj_gain_

    traj_loss_all = traj_loss + traj_loss_all
    traj_gain_all = traj_gain + traj_gain_all
    
    print(traj_loss_all)

    end_time = time.time()  # End timing
    print(f"Tile {ti} processed in {end_time - start_time:.2f} seconds.\n")

if(tiled and areaunit == 'perc_region'):
    traj_loss_all_ = 100*traj_loss_all/sum_rregion 
    traj_gain_all_ = 100*traj_gain_all/sum_rregion 
if(tiled and areaunit == 'perc_extent'):
    traj_loss_all_ = 100*traj_loss_all/sum_extent
    traj_gain_all_ = 100*traj_gain_all/sum_extent

gain_line = np.sum(np.sum(traj_gain_all))/np.sum(diff_years)
loss_line = np.sum(np.sum(traj_loss_all))/np.sum(diff_years)

components = [quantity_all,exchange_all,alternation_all]
if(annual is True):
    traj_loss_a = traj_loss_all_.div(diff_years, axis=0)  # Row-wise division
    traj_gain_a = traj_gain_all_.div(diff_years, axis=0)  # Row-wise division
    traj_result = traj_loss_a, traj_gain_a, gain_line, loss_line,components
if(annual is False):
    traj_result = traj_loss_all_, traj_gain_all_, gain_line,loss_line,components

print("Finished computing stats")
print(traj_loss_all)


#%% save traj
merged_traj = xr.combine_by_coords(traj_all).astype('int8')
merged_traj.chunk({"y": 10000, "x": 10000})
merged_traj.rio.to_raster(data_dir + 'traj_1.tif',
                          tile = True,
                          windowed = True,
                          blockxsize=10000,  # Set block size to match chunk size
                        blockysize=10000)


#%% save to csv 
writepath = data_dir
traj_loss_all.to_csv(data_dir + "af_loss_3.csv")
traj_gain_all.to_csv(data_dir + "af_gain_3.csv")

#%% 
import functions
import importlib
importlib.reload(functions)

functions.plot_traj_stack(traj_result,areaunit)
#com_perc = functions.components(reclassed_,ref_change,perc_unit = areaunit, type = type)
#functions.plot_comp(com_perc)

#%%
def get_components(components,type = 'raster',perc_unit = 'perc_region'):
    global netstatus
    diff_years = np.diff(np.array(years).astype(int))
    quantity, exchange, alternation = components[0],components[1],components[2]

    if (quantity>0):
        netstatus = 'gain'
    if (quantity<0):
        netstatus = 'loss'
        quantity = abs(quantity)
        components[0] = quantity

    if (perc_unit == 'perc_region'):
        com_perc =  [(float(x) * 100) / float(sum_rregion) for x in components]
    elif (perc_unit == 'perc_extent'):
        com_perc =  [(float(x) * 100) / float(sum_extent) for x in components]

    com_perc = [x/np.sum(diff_years) for x in com_perc]

    return com_perc

#%%
combined_da = split_tiles(combined_data)




# Generate masks for traj classes (0-8)
classes = da.arange(9)
traj_masks = da.equal(traj[None, :, :], classes[:, None, None])  # Shape: (9, y, x)

# Compute masks for -1 and 1 in ref_change
mask_loss = (ref_change == -1).astype("int8")  # Shape: (time, y, x)
mask_gain = (ref_change == 1).astype("int8")

# Reshape to 2D matrices and compute counts via matrix multiplication
mask_loss_flat = mask_loss.reshape(mask_loss.shape[0], -1)  # (time, y*x)
mask_gain_flat = mask_loss.reshape(mask_gain.shape[0], -1)  # (time, y*x)

traj_masks_flat = traj_masks.reshape(traj_masks.shape[0], -1)  # (9, y*x)

# Compute counts using Dask's dot product
counts_neg1 = traj_masks_flat.dot(mask_loss_flat.T)  # Shape: (9, time)
counts_1 = traj_masks_flat.dot(mask_gain_flat.T)



loss_counts = (
     da.where(ref_change == -1, 1, 0)  # Focus on loss events
    .groupby(traj)  # Group by trajectory class
    .sum(axis = (1,2))  # Sum over space
    .compute()  # Trigger computation once
)


#%% solution 1
# test result: not very efficient, lots of spillovers 

traj_mask = (ref_change == -1) * traj 
# Mask out unwanted values (traj == 7, 8, or 0)
valid_traj = da.where((traj_mask != 7) & (traj_mask != 8) & (traj_mask != 0), traj_mask, np.nan)  # Replace unwanted values with NaN

# Flatten spatial dimensions
reshaped = valid_traj.reshape(valid_traj.shape[0], -1)  # Shape: (36, Y*X)

# Unique values to count (1-6)
unique_values = da.arange(1, 7)

# Compute counts using a boolean mask instead of bincount
counts = da.vstack([(reshaped == val).sum(axis=1) for val in unique_values])

result = counts.T

# %%
