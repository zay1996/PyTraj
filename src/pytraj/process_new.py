from pytraj import functions_dask_new
from pytraj import functions
from pytraj import get_data



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
            write_dir = None,
            res = None,
            annual = True,
            split_flag = 'auto'):

    traj_results, com_perc = None, None
    if (type_ == 'raster'):
        raster_data,params = get_data.get_data(years, 
                pres_val, 
                nodata_val,
                filepath = filepath,
                type_ = type_)

        traj_init = functions_dask_new.TrajectoryAnalysis(
                                            raster_data, 
                                            params,
                                            type_ = type_,
                                            chunk_size = chunk_size,
                                            areaunit = areaunit, 
                                            weight = weight, 
                                            annual = annual,
                                            save_tif = save_tif,
                                            write_dir = write_dir,
                                            split_flag = split_flag)
        traj,traj_results = traj_init.process_data()
        traj_init.plot_traj_map(traj)
        traj_init.plot_traj_stack(traj_results)
        com_perc = traj_init.get_components(traj_results,areaunit)
        traj_init.plot_comp(com_perc)

        return traj_init,traj_results,com_perc
        
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


def adjust_graphs(results,areaunit,type_,traj_init = None,stack_kwargs = None, comp_kwargs = None):
    '''
    '''
    traj_results,com_perc = results
    if (traj_results is not None and stack_kwargs):
        if(type_ == 'smallraster' or type_ == 'table'):
            functions.plot_traj_stack(traj_results,areaunit,**stack_kwargs)
        elif(type_ == 'raster'):
            traj_init.plot_traj_stack(traj_results,**stack_kwargs)
    if (com_perc is not None and comp_kwargs):
        if(type_ == 'smallraster' or type_ == 'table'):
            functions.plot_comp(com_perc,areaunit,**comp_kwargs)
        elif(type_ == 'raster'):
            traj_init.plot_comp(com_perc,areaunit,**comp_kwargs)
