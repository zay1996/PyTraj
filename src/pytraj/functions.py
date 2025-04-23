import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from matplotlib import colors
import pandas as pd
import numpy as np
#from osgeo import gdal 
#from main import *


traj_list_all = ['No Data',
            'Loss without Alternation',
            'Gain without Alternation',
            'Loss with Alternation',
            'Gain with Alternation',
            'All Alternation Loss First',
            'All Alternation Gain First',
            'Stable Presence',
            'Stable Absence']



def unwrap_params(input_,params,type = 'raster'):
    global pres_val, nodata_val, years  # Declare variables as global
    global weight, areaunit, res
    pres_val = params['pres_val']
    nodata_val = params['nodata_val']
    years = params['years']
    areaunit = params['areaunit']
    res = params['res']


    if (type == 'raster'):
        global nr,nc
        nr = np.shape(input_)[1]
        nc = np.shape(input_)[2]
    if (type == 'table'):
        global ns
        ns = np.shape(input_)[0]
        weight = params['weight']
    global nt
    nt = len(years)

def reclass(input_,pres_val,nodata_val):
    
    reclassed = np.full_like(input_, fill_value=1, dtype="int8")

    # Apply conditions using Dask's `where()` (NumPy-compatible)
    reclassed = np.where(input_ == pres_val, 2, reclassed)  # Set 2 where combined_data == 3
    reclassed = np.where(input_ == nodata_val, 0, reclassed)  # Set 0 where combined_data == 0
    
    return reclassed


def get_traj(input_,type = 'raster'):
    # prepare change array
    print("identifying trajectories...")
    input_ar = reclass(input_,pres_val,nodata_val)
    if (type == 'table'):
        #input_ar = input_.iloc[:,1:-1].to_numpy().T
        ref_change = np.diff(input_ar,axis = 0)
        #adjacent_sum = ref_ar[:,:-1] + ref_ar[:,1:]
        adjacent_sum = input_ar[:-1,:] + input_ar[1:,:]

        traj = np.zeros((np.shape(input_)[-1]),dtype = int)


    elif (type == 'raster'):
        ref_change = np.diff(input_ar,axis = 0)
        #adjacent_sum = ref_ar[:,:-1] + ref_ar[:,1:]
        adjacent_sum = input_ar[:-1,:,:] + input_ar[1:,:,:]
        traj = np.zeros((nr,nc),dtype = int)

    ref_change[adjacent_sum == 2*2] = 2 # stable presence 
    ref_change[adjacent_sum == 1*2] = 0 # stable absence 

    # ref_change: 0 - stable absence, -1, loss, 1, gain, 2 stable presence

    # identify trajectories 

    # count number of gains and losses 
    num_gain = np.sum(ref_change == 1, axis = 0)
    num_loss = np.sum(ref_change == -1, axis = 0)
    num_pre = np.sum(ref_change == 2, axis = 0)
    num_abs = np.sum(ref_change == 0, axis = 0)
    #num_alt = (num_gain > 0) & (num_loss > 0)
    num_alt = np.minimum(num_gain,num_loss)
    first_loss = np.argmax(ref_change == -1, axis=0)
    first_loss[num_loss == 0] = nt # give it an out of bound number
    first_gain = np.argmax(ref_change == 1, axis = 0)
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
    traj[np.any(input_ar == 0, axis=0)] = 0

    print("Trajectories identified")
    return ref_change,traj,input_ar

#%% get annual change and percentage

def comp_change(input_ar,ref_change,traj,annual = True,weight = False):
    print("computing traj stats..")
    traj_list = traj_list_all[1:-2]
    traj_loss_ = pd.DataFrame(index=years[:-1], columns=traj_list)
    traj_gain_ = pd.DataFrame(index=years[:-1], columns=traj_list)
    traj_loss = traj_loss_.copy()
    traj_gain = traj_gain_.copy()
    traj_loss_a = traj_loss_.copy()
    traj_gain_a = traj_gain_.copy()
    diff_years = np.diff(np.array(years).astype(int))

    bichange = (input_ar[-1] - input_ar[0]).astype("int8")
    if(weight is False):
        bigain = (bichange == 1).sum()
        biloss = (bichange == -1).sum()
        totalgain = (ref_change == 1).sum()
        totalloss = (ref_change == -1).sum()
    
    if(weight is not False):
        bigain = np.sum((bichange == 1) * weight)
        biloss = np.sum((bichange == -1)*weight)
        totalgain = np.sum((ref_change == 1)*weight) # validate
        totalloss = np.sum((ref_change == -1)*weight)
    
    quantity = bigain - biloss
    exchange = min(bigain,biloss)
    totalchange = (totalgain + totalloss)
    alternation = totalchange - quantity-exchange 
    stats = [quantity,exchange,alternation]

    if(areaunit == 'perc_region'):
        if(weight is False):
            size_rregion = np.sum(np.any(input_ar == 2,axis = 0))
        if(weight is not False):
            size_rregion = np.sum(np.any(input_ar == 2,axis = 0)*weight)
        
        stats.append(size_rregion)
        #sum_rregion = sum_rregion+size_rregion

    if(areaunit == 'perc_extent'):
        if(weight is False):
            size_extent = np.sum(np.any(input_ar != 0,axis = 0))
        if(weight is not False):
            size_extent = np.sum(np.any(input_ar != 0,axis = 0)*weight)
        stats.append(size_extent)
        #sum_extent = sum_extent + size_extent 
    if(areaunit != 'perc_region' and areaunit != 'perc_extent'):
        stats.append(1)

    for i,t in enumerate(traj_list):
        change_i = ref_change[:,traj == i+1]
        if(weight is not False):
            weight_i = weight[traj == i+1]
            traj_loss_.iloc[:,i] = -1*np.sum((change_i == -1).astype(int) * weight_i,axis = 1)
            traj_gain_.iloc[:,i] = np.sum((change_i == 1).astype(int) * weight_i,axis = 1)
            
        elif(weight == False):
            traj_loss_.iloc[:,i] = -1* np.sum(change_i == -1, axis = 1)
            traj_gain_.iloc[:,i] = np.sum(change_i == 1, axis = 1)
        
        if(areaunit == 'perc_region'):
            # compute relevant region 
            #traj_loss_p = traj_loss_.copy()
            #traj_gain_p = traj_gain_.copy()
            traj_loss.iloc[:,i] = 100*traj_loss_.iloc[:,i]/size_rregion
            traj_gain.iloc[:,i] = 100*traj_gain_.iloc[:,i]/size_rregion
        if(areaunit == 'perc_extent'):

            traj_loss.iloc[:,i] = 100*traj_loss_.iloc[:,i]/size_extent
            traj_gain.iloc[:,i] = 100*traj_gain_.iloc[:,i]/size_extent        
        elif(areaunit == 'km2'):
            #traj_loss_km = traj_loss_.copy()
            #traj_gain_km = traj_gain_.copy()
            traj_loss.iloc[:,i] = (traj_loss_.iloc[:,i]*(res[0]*res[1]))/(1000**2)
            traj_gain.iloc[:,i] = (traj_gain_.iloc[:,i]*(res[0]*res[1]))/(1000**2)
            #traj_loss = traj_loss_km
            #traj_gain = traj_gain_km
        elif(areaunit == 'pixels'):
            traj_loss = traj_loss_
            traj_gain = traj_gain_

        
        #traj_loss_a.iloc[:,i] = traj_loss.iloc[:,i]/diff_years # get annual change
        #traj_gain_a.iloc[:,i] = traj_gain.iloc[:,i]/diff_years # get annual change 
    traj_loss_a = traj_loss.div(diff_years, axis=0)  # Row-wise division
    traj_gain_a = traj_gain.div(diff_years, axis=0)  # Row-wise division

    gain_line = np.sum(np.sum(traj_gain))/np.sum(diff_years)
    loss_line = np.sum(np.sum(traj_loss))/np.sum(diff_years)

    print("Finished computing stats")
    
    if(annual is True):
        return traj_loss_a, traj_gain_a, gain_line, loss_line,stats
    if(annual is False):
        return traj_loss, traj_gain, gain_line,loss_line,stats


#%% plot trajectories map
def plot_traj_map(traj):

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

    colorlist = ['white','#BF2024','#0F71B8','#EE1D23','#29ACE2','#C5C62E','#F1EB1A','Grey','Silver']

    cmap = colors.ListedColormap(colorlist)
    boundaries = np.arange(-0.5,9.5,1)
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)


    axarr.imshow(traj,cmap=cmap,norm=norm,interpolation='nearest')
    axarr.axis('off')
    plt.subplots_adjust(wspace=0.05, hspace=0.2)

    patches = [mpatches.Patch(color=colorlist[i], label=traj_list_map[i]) for i in np.arange(len(traj_list_map))]
    # put those patched as legend-handles into the legend
    axarr.legend(handles=patches,bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0. )
    
    f.tight_layout()
#f.suptitle('Water change based on direct change detection', fontsize=20)
#%% plot trajectories stacked bar
def plot_traj_stack(traj_result, areaunit, rotation = 90, ylim = None):
    
    traj_list = traj_list_all[1:-2]
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
        ax.set_ylabel("Annual loss and gain (% out of Union Presence)",fontsize = 16)
    if(areaunit == 'perc_extent'):
        ax.set_ylabel("Annual loss and gain (% out of spatial extent)", fontsize = 16)
                
                
    ax.set_xlabel('Time Interval',fontsize=20)
    
    ax.set_xticks(np.array(years).astype(int))  # Set the positions of the ticks
    ax.set_xticklabels(np.array(years).astype('str'),rotation = rotation)         # Set the labels for the ticks
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    #ax.figure.autofmt_xdate()
    ax.axhline(y=0,color='0',linewidth=0.5)
    # Add gross lines
    plt.axhline(y=gain_line, color='black', linestyle='--', linewidth=1.5, label='Gain Line')
    plt.axhline(y=loss_line, color='black', linestyle=':', linewidth=1.5, label='Loss Line')
        
    #ymin = 
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
def get_components(traj_results,type_ = 'raster',comp_unit = 'perc_region'):
    global netstatus
    _,_,_,_,components = traj_results
    diff_years = np.diff(np.array(years).astype(int))
    quantity, exchange, alternation,sum_region = components

    if (quantity>0):
        netstatus = 'gain'
    if (quantity<0):
        netstatus = 'loss'
        quantity = abs(quantity)
        components[0] = quantity

    com_perc =  [(float(x) * 100) / float(sum_region) for x in components]
    if (comp_unit == 'perc_region'):
        sum_rregion = sum_region
        com_perc =  [(float(x) * 100) / float(sum_rregion) for x in components]
    elif (comp_unit == 'perc_extent'):
        sum_extent = sum_region 
        com_perc =  [(float(x) * 100) / float(sum_extent) for x in components]
    elif(comp_unit == 'km2'):
        com_perc = [(float(x) * abs(res[0])*abs(res[1])) /(1000**2) for x in components] 
    
    com_perc = [x/np.sum(diff_years) for x in com_perc]

    return com_perc


#%% plot components
def plot_comp(com_perc,areaunit,ylim = None):
    fig, ax = plt.subplots(figsize=(10,10))

    #colorlist = ['brown','teal','coral','aquamarine','pink','purple']
    colorlist = ['black','grey','silver']
    com_list = ['Quantity ' + netstatus,'Exchange','Alternation']
    bottom = np.cumsum([0] + com_perc[:-1])
    width = 10
    for i in range(len(com_list)):
        ax.bar(0,com_perc[i],width=width,  color = colorlist[i], bottom = bottom[i], \
            align='edge', label = com_list[i])


                
    ax.set_xlabel('Time Interval',fontsize=20)

    if(areaunit == 'perc_region'):
        ax.set_ylabel('Annual Change (% of unified size)',fontsize=20)
    if(areaunit == 'perc_extent'):
        ax.set_ylabel('Annual Change (% of spatial extent)',fontsize=20)        
    if(areaunit == 'km2'):
        ax.set_ylabel('Annual Change (km²)',fontsize = 20)

    #ax.set_ylim([0,1.8])
    if (ylim is not None):
        ax.set_ylim(ylim)
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


def run_all(input_,params,areaunit = 'perc_region', type = 'raster',weight = False):
    '''
    
    '''
    unwrap_params(input_,params,type = type)
    ref_change,traj,input_ar = get_traj(input_,type)
    traj_result = comp_change(input_ar,ref_change, traj,areaunit = areaunit,weight = weight)
    if (type == 'raster'):
        plot_traj_map(traj)
    plot_traj_stack(traj_result,areaunit)
    com_perc = components(input_ar,ref_change,perc_unit = areaunit, type = type)
    plot_comp(com_perc)
# %%
