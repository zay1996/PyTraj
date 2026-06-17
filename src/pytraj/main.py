from pytraj import functions_dask_new
from pytraj import functions
from pytraj import get_data



def run_traj(filepath,
            years,
            pres_val,
            nodata_val,
            areaunit = 'perc_region',
            data_type = 'raster',
            weight = False,
            chunk_size = 1000,
            run_map = True,
            run_stacked = True,
            run_comp = True,
            run_pizza = True,
            export_map = None,
            res = None,
            annual = True,
            split_flag = 'auto',
            tile_row = None,
            tile_col = None):

    traj_results, com_perc = None, None
    if (data_type== 'raster'):
        #print("res in here is ",res)
        raster_data,params = get_data.get_data(years, 
                pres_val, 
                nodata_val,
                filepath = filepath,
                data_type = data_type,
                chunk_size = chunk_size,
                res = res)

        traj_init = functions_dask_new.TrajectoryAnalysis(
                                            raster_data, 
                                            params,
                                            data_type = data_type,
                                            chunk_size = chunk_size,
                                            areaunit = areaunit, 
                                            weight = weight, 
                                            annual = annual,
                                            export_map = export_map ,
                                            split_flag = split_flag,
                                            tile_row = tile_row,
                                            tile_col = tile_col)
        traj,traj_results = traj_init.process_data()
        traj_init.plot_traj_map(traj)
        traj_init.plot_traj_stack(traj_results)
        traj_init.plot_pizza(traj)
        com_perc = traj_init.get_components(traj_results,areaunit)
        traj_init.plot_comp(com_perc)

        traj_loss, traj_gain, gain_line,loss_line,stats = traj_results 
        outputs = {
        "traj": traj,
        "traj_loss": traj_loss,
        "traj_gain": traj_gain,
        "gainloss_line":[gain_line,loss_line],
        "components": com_perc
        }

        return outputs
        
    if(data_type == 'table' or data_type == 'smallraster'):
        input_,params = get_data.get_data(years,
                                          pres_val,
                                          nodata_val,
                                          filepath = filepath,
                                          areaunit = areaunit,
                                          data_type = data_type,
                                          weight = weight,
                                          res = res)
        functions.unwrap_params(input_,params,type = data_type)
        ref_change,traj,input_ar = functions.get_traj(input_,type = data_type)
        
        if (data_type== 'smallraster' and run_map is True):
            functions.plot_traj_map(traj)

        if (run_stacked is True):
            traj_results = functions.comp_change(input_ar,
                                                ref_change, 
                                                traj,
                                                weight = params['weight'])
            functions.plot_traj_stack(traj_results,areaunit)


        if (run_comp is True):
            com_perc = functions.get_components(traj_results,comp_unit = areaunit, type_ = data_type)
            functions.plot_comp(com_perc,areaunit)

        traj_loss, traj_gain, gain_line,loss_line,stats = traj_results 
        outputs = {
        "traj": traj,
        "traj_loss": traj_loss,
        "traj_gain": traj_gain,
        "gainloss_line":[gain_line,loss_line],
        "components": com_perc
        }

        return outputs


def adjust_graphs(results,areaunit,data_type,traj_init = None,stack_kwargs = None, comp_kwargs = None):
    '''
    '''
    traj_results,com_perc = results
    if (traj_results is not None and stack_kwargs):
        if(data_type == 'smallraster' or data_type == 'table'):
            functions.plot_traj_stack(traj_results,areaunit,**stack_kwargs)
        elif(data_type == 'raster'):
            traj_init.plot_traj_stack(traj_results,**stack_kwargs)
    if (com_perc is not None and comp_kwargs):
        if(data_type == 'smallraster' or data_type== 'table'):
            functions.plot_comp(com_perc,areaunit,**comp_kwargs)
        elif(data_type == 'raster'):
            traj_init.plot_comp(com_perc,areaunit,**comp_kwargs)
