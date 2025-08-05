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
import pytraj
from matplotlib_scalebar.scalebar import ScaleBar
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
#%%
class TrajectoryAnalysis:
    def __init__(self,
                 input_, 
                 params,
                 type_ = 'raster',
                 chunk_size = 1000,
                 areaunit = 'pixels', 
                 weight = False, 
                 annual = True,
                 save_tif = False,
                 write_dir = None,
                 split_flag = 'auto',
                 tile_row = None,
                 tile_col = None
                 ):
        
        self.input_ = input_
        self.params = params
        self.type_ = type_
        self.areaunit = areaunit
        self.weight = weight
        self.annual = annual 
        self.chunk_size = chunk_size
        self.save_tif = save_tif
        self.write_dir = write_dir
        self.split_flag = split_flag 
        self.tile_row = tile_row
        self.tile_col = tile_col 
        self.pres_val = self.params['pres_val']
        self.nodata_val = self.params['nodata_val']
        self.years = self.params['years']
        self.time_dim = self.params['time_dim']
        self.res = self.params['res']
        self.nt = self.params['nt']

        self.traj_list_all = ['No Data',
                    'Loss without Alternation',
                    'Gain without Alternation',
                    'Loss with Alternation',
                    'Gain with Alternation',
                    'All Alternation Loss First',
                    'All Alternation Gain First',
                    'Stable Presence',
                    'Stable Absence']


    def split_tiles(self,tile_row,tile_col):
        raster_data = self.input_
        # Get the number of rows (y dimension) and columns (x dimension)
        num_rows, num_cols = raster_data.sizes['y'], raster_data.sizes['x']
        
        # Define the size of each tile (5x5 grid)
        tile_height = num_rows // tile_row  # Rows per tile
        tile_width = num_cols // tile_col   # Columns per tile
        
        tile_idx = 0
        
        tile_datasets = []
        nodata_val = self.nodata_val
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
                    tile_datasets.append(tile_data) 
        
        return tile_datasets


    def if_split(self,split_flag = 'auto',tile_row = None,tile_col = None):
        raster_data = self.input_
        if (split_flag == 'auto'):
            # Get total system memory (bytes)
            total_memory = psutil.virtual_memory().total  
            num_cores = psutil.cpu_count(logical=False)
            # Get dataset size in bytes
            dataset_size = raster_data.nbytes  

            # Check if dataset is too large
            if dataset_size > 3 * total_memory:
                print("Dataset is too large, splitting into tiles...")

                # Compute ideal chunk size (~1/4 of total memory)
                target_tile_size = total_memory // (2*num_cores)

                # Get dataset dimensions
                dims = raster_data.sizes

                t_size,y_size,x_size = raster_data.shape[0],raster_data.shape[1],raster_data.shape[2]


                # Split across x and y while keeping time intact
                tile_side_length = int(np.sqrt(target_tile_size / t_size))  # Adjust for time

                tile_row = int(y_size/tile_side_length)
                tile_col = int(x_size/tile_side_length)

                print(f"Suggested number of tiles: {tile_row} x {tile_col}")
                tiled = True
                #return True, (tile_row, tile_col)
                tile_datasets = self.split_tiles(tile_row,tile_col)
                return tiled,tile_datasets
            
            print("Dataset fits in memory, no need to split.")
            tiled = False
            return tiled,raster_data 
        if (split_flag == 'yes'):
            tile_row = self.tile_row
            tile_col = self.tile_col
            if (tile_row is None or tile_col is None):
                raise ValueError("Must specified args tile_row and tile_col is split_flag is True!\
                                for automatic splitting, use split_flag = auto")
            else:
                tiled = True
                tile_datasets = self.split_tiles(tile_row,tile_col)
                return tiled,tile_datasets 
        if (split_flag == 'no'):
            tiled = False
            return tiled,raster_data 

    @staticmethod
    def reclass(input_,pres_val,nodata_val):
        reclassed = da.full_like(input_, fill_value=1,dtype = 'ubyte')

        # Apply conditions using Dask's `where()` (NumPy-compatible)
        reclassed = da.where(input_ == pres_val, 2, reclassed)  # Set 2 where combined_data == 3
        if(nodata_val is not None):
            reclassed = da.where(input_ == nodata_val, 0, reclassed)  # Set 0 where combined_data == 0
        if(nodata_val is None):
            reclassed = da.where(da.isnan(input_), 0, reclassed)

        mask = da.any(reclassed == 0, axis=0)  
        # Broadcast mask back to original shape
        mask_broadcasted = da.broadcast_to(mask, reclassed.shape)

        # Use da.where to conditionally set values to 0
        reclassed = da.where(mask_broadcasted, 0, reclassed)
        return reclassed

    @staticmethod
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
            mask = (input_ == 0).any(axis=0)  
            mask_broadcasted = np.broadcast_to(mask, change_ar.shape)  # shape: (4, H, W)

            change_ar[mask_broadcasted] = -2 # no data 
            return change_ar


    @staticmethod
    def get_traj(change_block,nt):
        if np.all(change_block == -2):  # Skip background chunks
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
            traj[np.any(change_block == -2, axis=0)] = 0 # set any no data in time series as 0 

            return traj
    


    def compute_traj(self,tiled = False):
        input_xr = self.input_
        traj_list = self.traj_list_all[1:-2]

        input_da = input_xr.data 
        xr_traj = xr.zeros_like(input_xr[0]).astype('int8')

        pres_val = self.pres_val
        nodata_val = self.nodata_val
        nt = self.nt
        years = self.years
        res = self.res
        weight = self.weight
        areaunit = self.areaunit

        chunking = (nt, self.chunk_size,self.chunk_size)
        reclassed_ = self.reclass(input_da,pres_val,nodata_val)   
        reclassed_ = reclassed_.rechunk(chunking)
        # Define the template (must match expected output shape)
        change_template = da.zeros_like(reclassed_[1:], dtype="int8")
        # Apply Dask map_blocks (directly on the Dask array)
        ref_change = da.map_blocks(
            self.get_change, 
            reclassed_, 
            dtype="int8",
            meta = change_template,
            new_axis = None,
            chunks =(reclassed_.chunks[0][0] - 1,) + reclassed_.chunks[1:])
            
        traj = da.map_blocks(
            self.get_traj,
            ref_change,
            nt,
            dtype = "int8",
            drop_axis = 0)
        
        xr_traj.data = traj
        #reclassed_ = da.where(traj == 0, 0,reclassed_) # reset no data as 0 
        #ref_change = da.where(traj == 0, 0,ref_change)

        # ------------------- Compute components ------------------- # 
        bichange = (reclassed_[-1] - reclassed_[0]).astype("int8")
        bigain = (bichange == 1).sum().compute()
        biloss = (bichange == -1).sum().compute()
        quantity = bigain - biloss
        exchange = 2*min(bigain,biloss)

        totalgain = (ref_change == 1).sum()
        totalloss = (ref_change == -1).sum()
        totalchange = (totalgain + totalloss).compute()
        alternation = totalchange - abs(quantity)-exchange 

        #print(f"quantity: {quantity}, exchange:{exchange},totalchange:{totalchange}")
        

        traj_loss_ = pd.DataFrame(index=years[:-1], columns=traj_list)
        traj_gain_ = pd.DataFrame(index=years[:-1], columns=traj_list)
        traj_loss = traj_loss_.copy()
        traj_gain = traj_gain_.copy()


        stats = [quantity,exchange,alternation]
        if(areaunit == 'perc_region'):
            
            size_rregion = np.sum(np.any(reclassed_ == 2,axis = 0)).compute()
            stats.append(size_rregion)
            #sum_rregion = sum_rregion+size_rregion


        if(areaunit == 'perc_extent'):
            size_extent = np.sum(np.any(reclassed_ != 0,axis = 0)).compute()
            stats.append(size_extent)
            #sum_extent = sum_extent + size_extent 
        if(areaunit != 'perc_region'  and areaunit != 'perc_extent'):
            stats.append(1)
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
                if(type(res) == int):
                    traj_loss.iloc[:,i] = (traj_loss_.iloc[:,i]*(abs(res)*abs(res)))/(1000**2)
                    traj_gain.iloc[:,i] = (traj_gain_.iloc[:,i]*(abs(res)*abs(res)))/(1000**2)
                else:
                    traj_loss.iloc[:,i] = (traj_loss_.iloc[:,i]*(abs(res[0])*abs(res[1])))/(1000**2)
                    traj_gain.iloc[:,i] = (traj_gain_.iloc[:,i]*(abs(res[0])*abs(res[1])))/(1000**2)                    
                #traj_loss = traj_loss_km
                #traj_gain = traj_gain_km
            elif(areaunit == 'pixels'):
                traj_loss = traj_loss_
                traj_gain = traj_gain_

        traj_outputs = traj_loss, traj_gain, stats, xr_traj 
        return traj_outputs


    def process_data(self,tile_row = None,tile_col = None):
        tiled,datasets = self.if_split(split_flag = self.split_flag,tile_row = tile_row,tile_col = tile_col)
        years = self.years
        annual = self.annual
        weight = self.weight
        chunk_size = self.chunk_size
        save_tif = self.save_tif
        areaunit = self.areaunit
        traj_list = self.traj_list_all[1:-2]
        write_dir = self.write_dir
        diff_years = np.diff(np.array(years).astype("int"))

        if (tiled is True):
            traj_loss_all = pd.DataFrame(index=years[:-1], columns=traj_list)
            traj_loss_all = traj_loss_all.fillna(0)
            traj_gain_all = pd.DataFrame(index=years[:-1], columns=traj_list)
            traj_gain_all = traj_gain_all.fillna(0)
            #global sum_rregion, sum_extent,quantity_all,exchange_all,alternation_all
            sum_rregion = 0
            sum_extent = 0
            sum_region = 0
            quantity_all,exchange_all,alternation_all = 0,0,0
            traj_all = []

            for ti in tqdm(range(len(datasets)), desc="Processing Tiles"):
                start_time = time.time()  # Start timing
                print(f"Processing tile {ti}...")
                input_xr = datasets[ti]
                traj_outputs = self.compute_traj(tiled = tiled)
                traj_loss, traj_gain, stats, xr_traj = traj_outputs 

                traj_loss_all = traj_loss + traj_loss_all
                traj_gain_all = traj_gain + traj_gain_all
                traj_all.append(xr_traj)
                quantity, exchange, alternation,size_region = stats
                if(areaunit == 'perc_region' or areaunit == 'perc_extent'):
                    sum_region += size_region
                if(areaunit != 'perc_region' and areaunit !='perc_extent'):
                    sum_region = 1 # repeated computation, revise?

                quantity_all += quantity
                exchange_all += exchange
                alternation_all += alternation
                
                end_time = time.time()  # End timing
                print(f"Tile {ti} processed in {end_time - start_time:.2f} seconds.\n")

            if(tiled and areaunit == 'perc_region'):
                sum_rregion = sum_region
                traj_loss_all_ = 100*traj_loss_all/sum_rregion 
                traj_gain_all_ = 100*traj_gain_all/sum_rregion 
            if(tiled and areaunit == 'perc_extent'):
                sum_extent = sum_region
                traj_loss_all_ = 100*traj_loss_all/sum_extent
                traj_gain_all_ = 100*traj_gain_all/sum_extent

            gain_line = np.sum(np.sum(traj_gain_all))/np.sum(diff_years)
            loss_line = np.sum(np.sum(traj_loss_all))/np.sum(diff_years)

            components = [quantity_all,exchange_all,alternation_all,sum_region]

            merged_traj = xr.combine_by_coords(traj_all).astype('int8')
            #merged_traj.chunk({"y": 10000, "x": 10000})
            if(save_tif is True):
                merged_traj.rio.to_raster(write_dir + 'traj_1.tif',
                                        tile = True,
                                        windowed = True,
                                        blockxsize=10000,  # Set block size to match chunk size
                                        blockysize=10000)
                
            if(annual is True):
                traj_loss_a = traj_loss_all_.div(diff_years, axis=0)  # Row-wise division
                traj_gain_a = traj_gain_all_.div(diff_years, axis=0)  # Row-wise division
                traj_result = traj_loss_a, traj_gain_a, gain_line, loss_line,components
                return merged_traj,traj_result
            if(annual is False):
                traj_result = traj_loss_all_, traj_gain_all_, gain_line,loss_line,components
                return merged_traj,traj_result 

            print("Finished computing stats")
            print(traj_loss_all)
            
        elif (tiled is False):
            start_time = time.time()  # Start timing
            print("identifying trajectories...")
            traj_outputs = self.compute_traj(
                                        tiled = tiled
                                        )
            traj_loss, traj_gain, components, xr_traj = traj_outputs 

            quantity, exchange, alternation,size_region = components
            #components = [quantity, exchange, alternation]
            gain_line = np.sum(np.sum(traj_gain))/np.sum(diff_years)
            loss_line = np.sum(np.sum(traj_loss))/np.sum(diff_years)

            if(save_tif is True):
                xr_traj.rio.to_raster(write_dir + 'traj_result.tif')
            print("Finished computing stats")
            end_time = time.time()  # End timing
            print(f"Data processed in {end_time - start_time:.2f} seconds.\n")
           
            if(annual is True):
                traj_loss_a = traj_loss.div(diff_years, axis=0)  # Row-wise division
                traj_gain_a = traj_gain.div(diff_years, axis=0)  # Row-wise division
                traj_result = traj_loss_a, traj_gain_a, gain_line, loss_line,components
                return xr_traj, traj_result
            if(annual is False):
                traj_result = traj_loss, traj_gain, gain_line,loss_line,components
                return xr_traj,traj_result


    def plot_traj_map(self,traj,north_arrow = True):

        f, axarr = plt.subplots(1,1,figsize=(10,10))

        traj_list_map = ['No Data',
                    'Loss without Alternation',
                    'Gain without Alternation',
                    'Loss with Alternation',
                    'Gain with Alternation',
                    'All Alternation Loss First',
                    'All Alternation Gain First',
                    'Stable Presence',
                    'Stable Absence']

        colorlist = ['white',"#A81A1F",'#0F71B8','#EE1D23','#29ACE2','#C5C62E','#F1EB1A','Grey','Silver']

        cmap = colors.ListedColormap(colorlist)
        boundaries = np.arange(-0.5,9.5,1)
        norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)


        axarr.imshow(traj,cmap=cmap,norm=norm,interpolation='nearest')
        axarr.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.2)
        res = self.res
        if res is not None:
            print("res map = ",res)
            font_prop = {'size':20}
            scalebar = ScaleBar(res, location='lower right',font_properties=font_prop)  # 1 pixel = 2 meter
            axarr.add_artist(scalebar)

        # Add north arrow SVG
        if north_arrow is True:
            if res is not None and res != 0:
                package_dir = os.path.dirname(os.path.abspath(pytraj.__file__)) # find directory of the package    
                north_arrow = package_dir+'/data/northarrow2.png'  # Update with the path to your SVG file
                img = Image.open(north_arrow)
                imagebox = OffsetImage(img, zoom=0.3)  # Adjust zoom as needed
                ab = AnnotationBbox(imagebox, (1.10, 0.1), frameon=False, xycoords='axes fraction', boxcoords="axes fraction", pad=0.0)
                axarr.add_artist(ab)

        patches = [mpatches.Patch(color=colorlist[i], label=traj_list_map[i]) for i in np.arange(len(traj_list_map))]
        # put those patched as legend-handles into the legend
        axarr.legend(handles=patches,bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0. )
        
        f.tight_layout()

    def plot_pizza(self,traj):
        import matplotlib
        traj_inds,traj_counts = da.unique(traj.data,return_counts = True)
        traj_counts = traj_counts.compute()
        traj_pizza = traj_counts[1:-1]
        traj_legend = self.traj_list_all[1:-1]
        traj_perc = traj_pizza/traj_pizza.sum()
        colorlist = ['#BF2024','#0F71B8','#EE1D23','#29ACE2','#C5C62E','#F1EB1A','Grey']


        # Function to display percentage labels
        def custom_autopct(pct):
            return f"{pct:.1f}%" if pct > 0 else ""

        plt.figure(figsize=(10, 8))  # Adjust figure size

        # Create the pie chart
        wedges, texts, autotexts = plt.pie(
            traj_perc,
            labels=None,  # No labels on the slices
            autopct=custom_autopct,
            colors=colorlist,
            startangle=90,
            #wedgeprops={'edgecolor': 'black'},
            textprops={'fontsize': 14},
            pctdistance=1.1,  # Moves the percentage text outward
            labeldistance=1.1  # Moves the label (percentages) outside the pie with connector lines
        )

        # Adjust the autopct text properties
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(12)
            autotext.set_bbox(dict(facecolor='none', edgecolor='none', alpha=0.6))  # Background for readability

        # Add a legend at the bottom with 3 columns
        plt.legend(wedges, traj_legend,  loc="lower center",
                bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=12, frameon=False)

        plt.title('PIE Trajectories Distribution', fontsize=20, pad=20)
        plt.show()
    #%% plot trajectories stacked bar
    def plot_traj_stack(self,traj_result,rotation = 90, ylim = None):
        
        nt = self.nt
        years = self.years
        traj_list = self.traj_list_all[1:-2]
        areaunit = self.areaunit
        traj_loss, traj_gain,gain_line,loss_line,_ = traj_result 
        
        # change the sequence to put all alt gain before all alt loss
        new_columns = traj_list[:-2] + traj_list[-1:] + traj_list[-2:-1]
        traj_loss = traj_loss[new_columns]
        traj_gain = traj_gain[new_columns]


        loss_bottom = traj_loss.cumsum(axis = 1)
        gain_bottom = traj_gain.cumsum(axis = 1)
        loss_bottom.insert(0, "0", np.zeros(nt-1))
        gain_bottom.insert(0,"0",np.zeros(nt-1))
        width=np.diff(np.array(years).astype(int))
        
        fig, ax = plt.subplots(figsize=(12,6))
        
        #colorlist = ['brown','teal','coral','aquamarine','pink','purple']
        colorlist = ['#BF2024','#0F71B8','#EE1D23','#29ACE2','#F1EB1A','#C5C62E']
        #colorlist = ['']
        
        for i in range(len(traj_list)):
            ax.bar(np.array(traj_gain.index).astype(int),traj_gain.iloc[:,i],width=width,bottom=gain_bottom.iloc[:,i], \
                color = colorlist[i], align='edge', label = new_columns[i])
            ax.bar(np.array(traj_loss.index).astype(int),traj_loss.iloc[:,i],width=width,bottom=loss_bottom.iloc[:,i], \
                color = colorlist[i], align='edge')

        if(areaunit == 'pixels'):
            ax.set_ylabel('Annual loss and gain (number of pixels)',fontsize=16)
        if(areaunit == 'sqm2'):
            ax.set_ylabel('Annual loss and gain (Square Meters)',fontsize = 16)
        if(areaunit == 'km2'):
            ax.set_ylabel('Annual loss and gain (km²)',fontsize = 16)
        if(areaunit == 'perc_region'):
            ax.set_ylabel("Annual loss and gain \n (% out of Union Presence)",fontsize = 16)
        if(areaunit == 'perc_extent'):
            ax.set_ylabel("Annual loss and gain \n (% out of spatial extent)", fontsize = 16)
                    
                    
        ax.set_xlabel('Time Interval',fontsize=20)
        
        ax.set_xticks(np.array(years).astype(int))  # Set the positions of the ticks
        ax.set_xticklabels(np.array(years).astype('str'),rotation = rotation)         # Set the labels for the ticks
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        #ax.figure.autofmt_xdate()
        ax.axhline(y=0,color='0',linewidth=0.5)
        # Add gross lines
        plt.axhline(y=gain_line, color='black', linestyle='--', linewidth=1.5, label='Gain Line')
        plt.axhline(y=loss_line, color='black', linestyle=':', linewidth=1.5, label='Loss Line')

        if(ylim is None):
            ax.set_ylim([np.min(np.min(loss_bottom))*1.1,np.max(np.max(gain_bottom))*1.1])
        if(ylim is not None):
            ax.set_ylim(ylim)
        #.axhline(y=gainline, color = 'black', linewidth = 2, label = 'Gain Line', linestyle = 'dashed')
        #ax.axhline(y=lossline, color = 'black', linewidth = 2, label = 'Loss Line', linestyle = 'dashdot')
        #plt.legend((p[0][0], p[1][0],p[2][0],p[3][0]), ('Disappearance','Split','Appearance','Coalescence')) 
        
        handles, labels = ax.get_legend_handles_labels()
        #order = np.arange(len(traj_list)+2)
        order = [0,1,2,3,4,5,7,6]

        ax.legend([handles[i] for i in order], [labels[i] for i in order], 
                ncol=4, 
                bbox_to_anchor=(0.5, -0.3), 
                frameon = False,
                loc='center', 
                fontsize = 15)
        
        
        plt.show()
        fig.tight_layout()
        

    #%% compute three components 
    def get_components(self,traj_results,comp_unit):
        res = self.res
        years = self.years
        _,_,_,_,components = traj_results
        diff_years = np.diff(np.array(years).astype(int))

        quantity, exchange, alternation,sum_region = components
        

        if (quantity>0):
            self.netstatus = 'gain'
        if (quantity<0):
            self.netstatus = 'loss'
            quantity = abs(quantity)
            components[0] = quantity
        if (quantity==0):
            self.netstatus = ''

        #com_perc =  [(float(x) * 100) / float(sum_region) for x in components]

        comp_unit = self.areaunit
        if (comp_unit == 'perc_region'):
            sum_rregion = sum_region
            com_perc =  [(float(x) * 100) / float(sum_rregion) for x in components]
        elif (comp_unit == 'perc_extent'):
            sum_extent = sum_region 
            com_perc =  [(float(x) * 100) / float(sum_extent) for x in components]
        elif(comp_unit == 'km2'):
            if(type(res) == int):
                com_perc = [(float(x) * abs(res)*abs(res)) /(1000**2) for x in components] 
            else: 
                com_perc = [(float(x) * abs(res[0])*abs(res[1])) /(1000**2) for x in components] 
        com_perc = [x/np.sum(diff_years) for x in com_perc]

        return com_perc

    #%% plot components
    def plot_comp(self,com_perc,ylim = None):
        fig, ax = plt.subplots(figsize=(10,10))
        areaunit = self.areaunit
        #colorlist = ['brown','teal','coral','aquamarine','pink','purple']
        colorlist = ['black','grey','silver']
        com_list = ['Quantity ' + self.netstatus,'Exchange','Alternation']
        bottom = np.cumsum([0] + com_perc[:-1])
        width = 10
        for i in range(len(com_list)):
            ax.bar(0,com_perc[i],width=width,  color = colorlist[i], bottom = bottom[i], \
                align='edge', label = com_list[i])
                    
        ax.set_xlabel('Time Interval',fontsize=20)

        if(areaunit == 'perc_region'):
            ax.set_ylabel('Annual Change (% of unified size)',fontsize=20)
        elif(areaunit == 'perc_extent'):
            ax.set_ylabel('Annual Change (% of spatial extent)',fontsize=20)        
        elif(areaunit == 'km2'):
            ax.set_ylabel('Annual Change (km²)',fontsize = 20)
        else:
            ax.set_ylabel(f'Annual Change ({areaunit})',fontsize = 20)

        if (ylim is not None):
            ax.set_ylim(ylim)
        #ax.set_ylim([0,1.8])
        handles, labels = ax.get_legend_handles_labels()
        ax.tick_params(axis='both', which='major', labelsize=15)
        order = np.arange(len(com_list))
        ax.legend([handles[i] for i in order], [labels[i] for i in order], 
                ncol=3, 
                bbox_to_anchor=(0.5, -0.1), 
                frameon = False,
                loc='center', 
                fontsize = 15)
        plt.xticks([])
        plt.show()
        fig.tight_layout()


