import functions_dask
import functions
import get_data




def run_traj(filepath,
            years,
            pres_val,
            nodata_val,
            areaunit = 'perc_region',
            type_ = 'raster',
            weight = False,
            chunk_size = 1000,
            run_map = True,
            run_stacked = True,
            run_comp = True,
            save_tif = False,
            res = None):
    
    global params 
    traj_results, com_perc = None, None
    if (type_ == 'raster'):
        raster_data,params = get_data.get_data(years, 
                pres_val, 
                nodata_val,
                filepath = filepath,
                type_ = type_)
        #functions_dask.get_params(params)

        tiled,datasets = functions_dask.if_split(raster_data)
        traj,traj_results = functions_dask.process_data(datasets,
                                                        years,
                                                        params,
                                                        areaunit=areaunit,
                                                        chunk_size = chunk_size,
                                                        weight = weight,
                                                        tiled = tiled,
                                                        save_tif = save_tif)
        functions_dask.plot_traj_map(traj)
        functions_dask.plot_traj_stack(traj_results,areaunit,params)
        com_perc = functions_dask.get_components(traj_results,params,comp_unit = areaunit)
        functions_dask.plot_comp(com_perc,areaunit)
        
    if(type_ == 'table' or type_ == 'smallraster'):
        input_,params = get_data.get_data(years,
                                          pres_val,
                                          nodata_val,
                                          filepath = filepath,
                                          areaunit = areaunit,
                                          type_ = type_,
                                          weight = weight,
                                          res = res)
        functions.unwrap_params(input_,params,type = type_)
        ref_change,traj,input_ar = functions.get_traj(input_,type = type_)
        
        if (type_ == 'smallraster' and run_map is True):
            functions.plot_traj_map(traj)

        if (run_stacked is True):
            traj_results = functions.comp_change(input_ar,
                                                ref_change, 
                                                traj,
                                                weight = params['weight'])
            functions.plot_traj_stack(traj_results,areaunit)


        if (run_comp is True):
            com_perc = functions.get_components(traj_results,comp_unit = areaunit, type_ = type_)
            functions.plot_comp(com_perc,areaunit)
        
    return traj_results, com_perc


def adjust_graphs(results,areaunit,type_,stack_kwargs = None, comp_kwargs = None):
    '''
    '''
    traj_results,com_perc = results
    if (traj_results is not None and stack_kwargs):
        if(type_ == 'smallraster' or type_ == 'table'):
            functions.plot_traj_stack(traj_results,areaunit,**stack_kwargs)
        elif(type_ == 'raster'):
            functions_dask.plot_traj_stack(traj_results,areaunit,**stack_kwargs)
    if (com_perc is not None and comp_kwargs):
        if(type_ == 'smallraster' or type_ == 'table'):
            functions.plot_comp(com_perc,areaunit,**comp_kwargs)
        elif(type_ == 'raster'):
            functions_dask.plot_comp(com_perc,areaunit,**comp_kwargs)
