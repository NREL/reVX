# Least cost transmission paths
Determine least cost transmission paths from possible wind and solar farms (supply curve (SC) points) to the electrical grid. Available components of the electical grid are substations, transmission lines, load centers and infinite sinks. The code only attempts to connect to a point on the transmission line closest to the SC point.

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
* `least_cost_xmission_cli.py` - TODO
* 

### Other notable Python files
* `trans_cap_costs.py` 
	* `TieLineCosts` - Determine least cost paths and transmission costs from SC point to multiple transmission grid elements using `skimage.graph.MCP_Geogetric` 
	* `TransCapCosts` - Determine total transmission cost including line cost and any substation construction or improvements.
	

* `least_cost_xmission.py` - Calculate costs from SC points to transmission features. By default, all SC points are used or a subset may be specified by GID.

* `least_cost_paths.py` - Parent class for `least_cost_ximssion.py`. I don't know how much of this is actually used. 
	
TODO
### Required data files
TODO 

## Workflow
TODO

## AOSWT utilities
The file `aoswt_utilities.py` contains utilitiy functions for preparing friction, barrier, and transmission features for the AOSWT analysis. Example Jupyter notebooks for these functions can be found in the `examples/least_cost_paths` directory of this repo.

## Examples
TODO: add cost calculator examples

Find least cost paths, costs, and connection costs on eagle login node for 1000MW capacity, saving results in current directory. Don't do this, it will overload the login node:

```
python least_cost_xmission_cli.py local \
--cost_fpath /shared-projects/rev/exclusions/xmission_costs.h5 \
--features_fpath /projects/rev/data/transmission/shapefiles/conus_allconns.gpkg \
--capacity_class 1000
```


```
python least_cost_xmission_cli.py local \
--cost_fpath /shared-projects/rev/transmission_tables/least_cost/offshore/aoswt_costs.h5 \
--features_fpath /shared-projects/rev/transmission_tables/least_cost/offshore/aoswt_pois.gpkg \
--capacity_class 100
```
```
python least_cost_xmission_cli.py local -v \
--cost_fpath /shared-projects/rev/transmission_tables/least_cost/offshore/aoswt_costs.h5 \
--features_fpath /shared-projects/rev/transmission_tables/least_cost/offshore/aoswt_pois.gpkg \
--capacity_class 100 \
--max_workers 1 \
--sc_point_gids [36092,36093]
```
### Process select SC points for AOSWT on Eagle
```
python -m reVX.least_cost_xmission.least_cost_xmission_cli from-config \
--config ~/reVX/reVX/least_cost_xmission/config/example_aoswt_eagle_config_debug.json \
--sc_point_gids [40139, 97919, 50000, 60000]
```
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
### Run AOSWT on a debug node
The below config file processes a single SC point (gid=40139) on a debug node. It also uses the optional `save_paths` and `radius` options to save the least coasts paths to a geopackage and force a cost raster clipping radius of 5000 pixels, versus determining the radius from the nearest sinks. Since this is an offshore analysis, the resolution SC point resolution is set to 118. The value for `allocation` should be set to the desired SLURM allocation.

```
{
  "execution_control": {
    "allocation": "YOUR_ALLOCATION",
    "feature": "-p debug",
    "memory": 178,
    "nodes": 20,
    "option": "eagle",
    "sites_per_worker": 100,
    "walltime": 1,
    "max_workers": 30
  },
  "cost_fpath": "/shared-projects/rev/transmission_tables/least_cost/offshore/aoswt_costs.h5",
  "features_fpath": "/shared-projects/rev/transmission_tables/least_cost/offshore/aoswt_pois.gpkg",
  "capacity_class": "100",
  "barrier_mult": "100",
  "log_directory": "/scratch/mbannist/lcp/test",
  "log_level": "DEBUG",
  "sc_point_gids": [40139],
  "resolution": 118,
  "save_paths": true,
  "radius": 5000
}
```

Assuming the above config file is saved as `config_debug.json` in the current directory, it can be kicked off with:

```
python -m reVX.least_cost_xmission.least_cost_xmission_cli from-config \
--config ./config_debug.json
```