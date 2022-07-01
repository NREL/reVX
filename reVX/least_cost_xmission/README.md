# Least cost transmission paths
Determine least cost transmission paths from possible wind and solar farms (supply curve (SC) points) to the electrical grid. Available components of the electical grid are substations, transmission lines, load centers and infinite sinks. The code only attempts to connect to a point on the transmission line closest to the SC point. This was initially used for land-based analyses, but has been modified for off-shore transmission as well.

TODO - ISOs

TODO - search radius

TODO - voltage class restrictions

## Costs
TODO

### Transmission costs
TODO
### Connection costs
TODO 
## Files
### Python command line interface (CLI) files
* `cost_creator_cli.py` - Compute transmission cost raster and save as hdf5. This includes creating slope and land use cost multipliers from source data and adding base transmission line construction costs. Final cost raster consists of line construction costs with all multipliers by ISO. 
* `least_cost_paths_cli.py` - TODO
* `least_cost_xmission_cli.py` - Calculate least cost paths, transmission paths, and connection costs. 

### Other notable Python files
* `trans_cap_costs.py` - Determine paths and costs for a single SC point
	* `TieLineCosts` - Determine least cost paths and transmission costs from SC point to multiple transmission grid elements using `skimage.graph.MCP_Geogetric` 
	* `TransCapCosts` - Determine total transmission cost including line cost and any substation construction or improvements.
* `least_cost_xmission.py` - Calculate costs from SC points to transmission features. By default, all SC points are used or a subset may be specified by GID.
* `least_cost_paths.py` - Parent class for `least_cost_xmission.py`. I don't know how much of this is actually used. 
* `aoswt_utilities.py` - Utilitiy functions and classes for preparing friction, barrier, and transmission features for the AOSWT analysis. Example Jupyter notebooks for these functions can be found in the `examples/least_cost_paths` directory of this repo.
	
### Required data files
TODO 

## Workflow
TODO



## CONUS (on-shore) Examples
TODO: add cost calculator examples
### Find CONUS least cost paths on a local eagle node
Find least cost paths, costs, and connection costs on eagle login node for 1000MW capacity and all SC points, saving results in current directory. These examples will overload the login nodes and should be run on a debug node.

```
python least_cost_xmission_cli.py local \
--cost_fpath /shared-projects/rev/exclusions/xmission_costs.h5 \
--features_fpath /projects/rev/data/transmission/shapefiles/conus_allconns.gpkg \
--capacity_class 1000
```

### Run on-shore analysis from a config file
The below file can be used to start a full CONUS analysis for the 1000MW power class. The setting `nodes` in `execution_control` will split the processing across five eagle nodes. 

```
{
  "execution_control": {
    "allocation": "YOUR_SLURM_ALLOCATION",
    "feature": "--qos=high",
    "memory": 178,
    "nodes": 5,
    "option": "eagle",
    "walltime": 4
  },
  "cost_fpath": "/shared-projects/rev/exclusions/xmission_costs.h5",
  "features_fpath": "/projects/rev/data/transmission/shapefiles/conus_allconns.gpkg",
  "capacity_class": "1000",
  "barrier_mult": "100",
  "log_directory": "/scratch/USER_NAME/log",
  "log_level": "INFO"
}
```

Assuming the above config file is saved as `config_conus.json` in the current directory, it can be kicked off with:

```
python -m reVX.least_cost_xmission.least_cost_xmission_cli from-config \
--config ./config_conus.json
```

## Atlantic Off-Show Wind Transmission (AOSWT) Examples
### Build friction and barriers layer
An example Jupyter notebook for building the friction and barrier layers can be found in the `examples/least_cost_paths` directory of this repo.

### Locally run a AOSWT analysis for a single SC point, plot the results, and save to a geopackage
This example uses `contextily` to add a base map to the plot, but is not required. AOSWT needs an aggregation "resolution" of 118. 

```
import contextily as cx
from rex.utilities.loggers import init_mult
from reVX.least_cost_xmission.least_cost_xmission import LeastCostXmission

# Start the logger
log_modules = [__name__, 'reVX', 'reV', 'rex']
init_mult('run_aoswt', '.', modules=log_modules, verbose=True)

cost_fpath = '/shared-projects/rev/transmission_tables/least_cost/offshore/aoswt_costs.h5'
features_fpath = '/shared-projects/rev/transmission_tables/least_cost/offshore/aoswt_pois.gpkg'
sc_point_gid = 51085

# Calculate paths
lcx = LeastCostXmission(cost_fpath, features_fpath, resolution=118)
paths = lcx.process_sc_points('100', sc_point_gids=[sc_point_gid], save_paths=True, 
                               max_workers=1, radius=5000)

# Plot the paths                                    
paths = paths.to_crs(epsg=3857)
ax = paths.plot(figsize=(20,20), alpha=0.5, edgecolor='red')
cx.add_basemap(ax, source=cx.providers.Stamen.TonerLite)

# Save to a geopackage
paths.to_file('example.gpkg', driver='GPKG')
```

### Find AOSWT least cost paths on a local eagle node
Find least cost paths, costs, and connection costs on eagle login node for 100MW capacity, saving results in current directory. These examples will overload the login nodes and should be run on a debug node.

```
python least_cost_xmission_cli.py local \
--cost_fpath /shared-projects/rev/transmission_tables/least_cost/offshore/aoswt_costs.h5 \
--features_fpath /shared-projects/rev/transmission_tables/least_cost/offshore/aoswt_pois.gpkg \
--capacity_class 100
```
Run the above analysis for only two SC points, using only one core.

```
python least_cost_xmission_cli.py local -v \
--cost_fpath /shared-projects/rev/transmission_tables/least_cost/offshore/aoswt_costs.h5 \
--features_fpath /shared-projects/rev/transmission_tables/least_cost/offshore/aoswt_pois.gpkg \
--capacity_class 100 \
--max_workers 1 \
--sc_point_gids [36092,36093]
```

### Run AOSWT from a config file
Using a config file is the prefered method of running an analysis. The below file processes a single SC point (gid=40139) on a debug node. Note that SLURM high quality of service on a standard node can be requested with `"feature": "--qos=high"`. This file also uses the optional `save_paths` and `radius` options to save the least coasts paths to a geopackage and force a cost raster clipping radius of 5000 pixels, versus determining the radius from the nearest sinks. Since this is an offshore analysis, the resolution SC point resolution is set to 118. The value for `allocation` should be set to the desired SLURM allocation. The `max_workers` key can be used to reduce the workers on each node if memory issues are encountered, but can typically be left out.

```
{
  "execution_control": {
    "allocation": "YOUR_SLURM_ALLOCATION",
    "feature": "-p debug",
    "memory": 178,
    "nodes": 2,
    "option": "eagle",
    "walltime": 1,
    "max_workers": 36
  },
  "name": "test", 
  "cost_fpath": "/shared-projects/rev/transmission_tables/least_cost/offshore/aoswt_costs.h5",
  "features_fpath": "/shared-projects/rev/transmission_tables/least_cost/offshore/aoswt_pois.gpkg",
  "capacity_class": "100",
  "barrier_mult": "100",
  "log_directory": "/scratch/USER_NAME/log",
  "log_level": "DEBUG",
  "sc_point_gids": [40139],
  "resolution": 118,
  "save_paths": true,
  "radius": 5000
}
```

Assuming the above config file is saved as `config_aoswt.json` in the current directory, it can be kicked off with:

```
python -m reVX.least_cost_xmission.least_cost_xmission_cli from-config \
--config ./config_aoswt.json
```

### Post processing
Running an analysis on multiple nodes will result in multiple output files. These can be collected via several means. The below command will combine all output files into a single geopackage, assuming `save_paths` was enabled. If paths are not saved, the output will consist of multiple CSV files that must be merged manually. 

```
python -m reVX.least_cost_xmission.least_cost_xmission_cli merge-output \
--out-file combined.gpkg \
output_files_*.gpkg
```

Transmission feature categories that are not desired in the final output can be dropped with:

```
python -m reVX.least_cost_xmission.least_cost_xmission_cli merge-output \
--out-file combined.gpkg \
--drop TransLine --drop LoadCen \
output_files_*.gpkg
```

Additionally, the results may be split into GeoJSONs by transmission feature connected with the following. This will not create a combined GeoPackage file. The optional `--simplify-geo` flag is also used. By default, the paths contain a vertex every 90m which results in large output files. Simplifying the geometries reduces the file size and creates more realistic paths. `simplify_geo` can also be set in the processing config file if desired. 

```
python -m reVX.least_cost_xmission.least_cost_xmission_cli merge-output \
--drop TransLine --drop LoadCen \
--split-to-geojson --out-path ./out \
--simplify-geo 100 \
output_files_*.gpkg
```