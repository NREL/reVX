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
