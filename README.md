PyTraj
================================================================================================
PyTraj is a Python implementation of the trajectory analysis method originally developed by Bilintoh et al. (2025) in R (https://github.com/bilintoh/timeseriesTrajectories). 

# 1. Installation 

Install the package using the following command:
```
pip install git+https://github.com/zay1996/PyTraj.git
```
# 2. Running the package 
-----------------------
The current version of PyTraj can be run using command-line with a Python interpreter.

 The script 'test.py` under the tests folder gives an example of running PyTraj in a script.

## Input Parameters
```python
Parameters
----------
filepath : str
    Path to the input data. For raster input, this can be a folder containing
    the input maps or the path to a raster file.

years : list of int
    Years corresponding to the input maps, provided in temporal order.
    Example: [2010, 2012, 2014, 2016, 2018, 2021]

pres_val : int
    Pixel value representing the presence category of interest.

nodata_val : int
    NoData value in the input maps.

areaunit : {"perc_region", "perc_extent", "km2", "pixels"}, default="perc_region"
    Area unit used for reporting gain and loss results.
    Options are:
    - "perc_region": reports area as a percentage of the relevant region.
    - "perc_extent": reports area as a percentage of the full spatial extent.
    - "km2": reports area in square kilometers, calculated from the map resolution.
    - "pixels": reports area as the number of pixels.
    The default is "perc_region".

data_type : {"raster", "smallraster","table"}, default="raster"
    Type of input data and processing mode. Use "raster" for Dask-based
    chunked processing of large raster datasets. Use "smallraster" to read
    the full raster into memory using NumPy, which is suitable only for
    smaller datasets. Use "table" if data is tabular and stored in csv. 

chunk_size : int, default=1000
    Chunk size used for Dask-based raster processing. This argument is used
    when data_type="raster".

run_map : bool, default=True
    Whether to generate trajectory map outputs.

run_stacked : bool, default=True
    Whether to generate stacked bar chart outputs.

run_comp : bool, default=True
    Whether to generate change component summary graphics.

export_map : str or None, default=None Output path prefix for exporting trajectory maps. 
The path should include the directory and output file name, but not the ".tif" extension. 
If None, maps are not exported. Example: export_map = r"D:\analysis\traj_map"

res : int or float, optional
    Spatial resolution of the input maps in meters. If not specified, the
    resolution is read automatically from the input raster when possible.

split_flag : {"auto", "yes", "no"}, default="auto"
    Controls whether the raster is split into tiles before processing.
    Use "auto" to let the program determine whether tiling is needed.
    Use "yes" to split the raster using the number of tile rows and columns
    specified by tile_row and tile_col. Use "no" to process the raster without
    tiling.

tile_row : int, optional
    Number of tile rows used when split_flag="yes".

tile_col : int, optional
    Number of tile columns used when split_flag="yes".

```
## Outputs

`run_traj()` returns a dictionary containing the trajectory map and summary outputs:

```python
outputs = {
    "traj": traj,
    "traj_loss": traj_loss,
    "traj_gain": traj_gain,
    "gainloss_line": [gain_line, loss_line],
    "components": com_perc,
}
```

Access each output by its key:

```python
outputs = run_traj(...)

traj = outputs["traj"]
traj_loss = outputs["traj_loss"]
traj_gain = outputs["traj_gain"]
gain_line, loss_line = outputs["gainloss_line"]
components = outputs["components"]
```

### `traj`

Trajectory map generated from the input map time series. Each pixel value represents the temporal trajectory of the presence category across all input dates.

The values in the trajectory map are coded as follows:

| Value | Trajectory class                 |
| ----: | -------------------------------- |
|     0 | NoData                           |
|     1 | Loss without alternation         |
|     2 | Gain without alternation         |
|     3 | Loss with alternation            |
|     4 | Gain with alternation            |
|     5 | All alternation loss first |
|     6 | All alternation gain first |
|     7 | Stable presence                  |
|     8 | Stable absence                   |


### `traj_loss`

A pandas DataFrame summarizing interval-based loss by trajectory type. 

### `traj_gain`

A pandas DataFrame summarizing interval-based gain by trajectory type. 

### `gainloss_line`

A list containing two outputs: `gain_line` and `loss_line`. These summarize gain and loss averaged across the entire time frame and are used to represent overall gain and loss trends through time.

### `components`

A pandas DataFrame or array containing the proportions of the three change components. This output summarizes the relative contribution of each change component to the overall trajectory pattern.

# References 
Bilintoh TM, Pontius RG, Zhang A. Methods to compare sites concerning a category’s change during various time intervals. GIScience & Remote Sensing, 2024, 61(1): 2409484. https://doi.org/10.1080/15481603.2024.2409484

# Acknowledgements
This work is supported by the National Science Foundation's grant OCE-2224608 entitled “LTER: Plum Island Ecosystems, the impact of changing landscapes and climate on interconnected coastal ecosystems”. 
