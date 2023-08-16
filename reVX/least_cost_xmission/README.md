# Least Cost Transmission Paths
Determine least cost transmission paths from possible wind and solar farms (supply curve (SC) points) to the electrical grid. Available components of the electrical grid are substations, transmission lines, load centers and infinite sinks. The code only attempts to connect to a point on the transmission line closest to the SC point. This was initially used for land-based analyses, but has been modified for offshore transmission as well.

<br>

## Files
### Python command-line interface (CLI) files
* [`cost_creator_cli.py`](cost_creator_cli.py) - Compute transmission cost raster and save as hdf5. This includes creating slope and land use cost multipliers from source data and adding base transmission line construction costs. Final cost raster consists of line construction costs with all multipliers by ISO.
* [`least_cost_paths_cli.py`](least_cost_paths_cli.py) - Calculate least cost paths between a set of points.
* [`least_cost_xmission_cli.py`](least_cost_xmission_cli.py) - Calculate least cost transmission paths and connection costs.

### Other notable Python files
* [`trans_cap_costs.py`](trans_cap_costs.py) - Determine paths and costs for a single SC point
	* `TieLineCosts` - Determine least cost paths and transmission costs from SC point to multiple transmission grid elements using [`skimage.graph.MCP_Geogetric`](https://scikit-image.org/docs/stable/api/skimage.graph.html#mcp-geometric)
	* `TransCapCosts` - Determine total transmission cost including line cost and any substation construction or improvements.
* [`least_cost_xmission.py`](least_cost_xmission.py) - Calculate costs from SC points to transmission features. By default, all SC points are used or a subset may be specified by GID.
* [`least_cost_paths.py`](least_cost_paths.py) - Parent class for `least_cost_xmission.py`.
* [`aoswt_utilities.py`](aoswt_utilities.py) - Utility functions and classes for preparing friction, barrier, and transmission features for the AOSWT analysis. Example Jupyter notebooks for these functions can be found in the [`examples/least_cost_paths`](../../examples/least_cost_paths/) directory of this repository.

<br>

# CONUS (Onshore) Examples
## Costs
The below file can be used as a template to compute the costs to be used in a Least Cost Path analysis described in more detail below.
```
{
  "execution_control": {
    "allocation": "YOUR_SLURM_ALLOCATION",
    "feature": "--qos=normal",
    "memory": 178,
    "option": "eagle",
    "walltime": 4
  },
  "h5_fpath": "/path/to/output/h5/file/that/already/contains/NLCD/and/slope/layers.h5",
  "iso_regions": "/path/to/ISO/regions/raster.tiff",
  "excl_h5": "/path/to/exclusion/file/with/NLCD/and/slope/layers.h5",
  "log_directory": "/scratch/USER_NAME/log",
  "log_level": "INFO"
}
```
See [`cost_creator_cli.local`](cost_creator_cli.py) for more info about these inputs. Your cost H5 file output should look something like this:
```
ISO_regions              Dataset {1, 33792, 48640}
latitude                 Dataset {33792, 48640}
longitude                Dataset {33792, 48640}
srtm_slope               Dataset {1, 33792, 48640}
tie_line_costs_102MW     Dataset {1, 33792, 48640}
tie_line_costs_1500MW    Dataset {1, 33792, 48640}
tie_line_costs_205MW     Dataset {1, 33792, 48640}
tie_line_costs_3000MW    Dataset {1, 33792, 48640}
tie_line_costs_400MW     Dataset {1, 33792, 48640}
tie_line_multipliers     Dataset {1, 33792, 48640}
transmission_barrier     Dataset {1, 33792, 48640}
usa_mrlc_nlcd2011        Dataset {1, 33792, 48640}
```

<br>


### Find CONUS least cost paths on a local eagle node
Find least cost paths, costs, and connection costs on eagle login node for 1000MW capacity and all SC points, saving results in current directory. These examples will overload the login nodes and should be run on a debug node.

```
python least_cost_xmission_cli.py local \
--cost_fpath /shared-projects/rev/exclusions/xmission_costs.h5 \
--features_fpath /projects/rev/data/transmission/shapefiles/conus_allconns.gpkg \
--capacity_class 1000
```

### Run onshore analysis from a config file
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

<br>

# Reinforced Transmission
In this methodology, total interconnection costs are comprised of two components: *point-of-interconnection costs* and *network upgrade costs*. Point-of-interconnection costs include the cost of the spur line between the RE facility (SC point) and the connected substation, as well as the cost to upgrade the substation itself. Network upgrade costs are represented by costs to increase the transmission capacity between the connected substation and the main network node in the balancing area. Network upgrade costs are assumed to be 50% of the cost of a new greenfield transmission of the same voltage as the existing transmission lines along the same path. The 50% heuristic represents cost for reconductoring or increasing the number of circuits along those lines. `reVX` (and any derivative products, e.g. `reV`) do not include estimates for longer-distance transmission that may be needed to export the renewable energy between one balancing area and another.

### Key assumptions under this approach:

- Area of interest (e.g. CONUS) is broken up into "Balancing Areas"
- Each Balancing Area has exactly one "Network Node" (typically a city or highly populated area)
- SC points **may only connect to substations**
- Substations that a SC point connects to **must be in the same Balancing Area as the SC point**
    - This assumption can be relaxed to allow connections within the same state.
- Reinforcement costs are calculated based on the distance between the substation a SC point connected to and the Network Node in that Balancing Area
    - The path used to calculate reinforcement costs is traced along existing transmission lines **for as long as possible**.
    - The reinforcement cost is taken to be half (50%) of the total greenfield cost of the transmission line being traced. If a reinforcement path traces along multiple transmission lines, the corresponding greenfield costs are used for each segment. If multiple transmission lines are available in a single raster pixel, the cost for the highest-voltage line is used. Wherever there is no transmission line, a default greenfield cost assumption (specified by the user; typically 230 kV) is used.

An example plot of this method is shown below (credit: Anthony Lopez):

![sample_r_im](../../examples/least_cost_paths/sample_r_im.png "Sample Reinforcement Results")

In this plot, (light) grey lines represent existing transmission, orange lines represent spur lines built from a supply curve point to a substation within the same Balancing Area, and green lines represent the reinforcement path calculated from the substation the Network Node (red dot in the center).


## Calculating Reinforced Transmission Tables

First, map the substations in your data set to the balancing areas using the following reVX command:

    least-cost-paths map-ba --features_fpath /projects/rev/data/transmission/shapefiles/conus_allconns.gpkg --balancing_areas_fpath /shared-projects/rev/transmission_tables/reinforced_transmission/data/ReEDS_BA.gpkg --out_file substations_with_ba.gpkg


Next, compute the reinforcement paths on multiple nodes. Use the file below as a template (`reinforcement_path_costs_config.json`):

```
{
    "execution_control": {
      "allocation": "YOUR_SLURM_ALLOCATION",
      "feature": "--qos=normal",
      "memory": 178,
      "option": "eagle",
      "max_workers": 1,
      "nodes": 10,
      "walltime": 1
    },
    "cost_fpath": "/shared-projects/rev/exclusions/xmission_costs.h5",
    "features_fpath": "/shared-projects/rev/transmission_tables/reinforced_transmission/reinforcement_costs/substations_with_ba.gpkg",
    "network_nodes_fpath": "/shared-projects/rev/transmission_tables/reinforced_transmission/data/transmission_endpoints",
    "transmission_lines_fpath": "/projects/rev/data/transmission/shapefiles/conus_allconns.gpkg",
    "capacity_class": "400",
    "barrier_mult": "100",
    "log_directory": "./logs",
    "log_level": "INFO",
}
```

Note that we are specifying ``"capacity_class": "400"``  to use the 230 kV (400MW capacity) greenfield costs for portions of the reinforcement paths that do no have existing transmission. If you would like to save the reinforcement path geometries, simply add `"save_paths": true` to the file, but note that this may increase your data product size significantly. If you would like to allow substations to connect to endpoints within the same state, add `"allow_connections_within_states": true` to the file.

After putting together your config file, simply call

    least-cost-paths from-config -c reinforcement_path_costs_config.json

This will generate 10 chunked files (since we used 10 nodes in the config above). To merge the data, simply call

    least-cost-xmission merge-output -of reinforcement_costs_400MW_230kV.gpkg -od /shared-projects/rev/transmission_tables/reinforced_transmission/reinforcement_costs reinforcement_costs_*_400MW_230kV.csv

You should now have a file containing all of the reinforcement costs for the substations in your dataset. Next, compute the spur line transmission costs for these substations using the following template config (`least_cost_transmission_1000MW.json`):

```
{
    "execution_control": {
      "allocation": "YOUR_SLURM_ALLOCATION",
      "feature": "--qos=normal",
      "memory": 500,
      "nodes": 100,
      "option": "eagle",
      "max_workers": 36,
      "walltime": 1
    },
    "cost_fpath": "/shared-projects/rev/exclusions/xmission_costs.h5",
    "features_fpath": "/shared-projects/rev/transmission_tables/reinforced_transmission/reinforcement_costs/substations_with_ba.gpkg",
    "balancing_areas_fpath": "/shared-projects/rev/transmission_tables/reinforced_transmission/data/ReEDS_BA.gpkg",
    "capacity_class": "1000",
    "barrier_mult": "100",
    "log_directory": "./logs",
    "log_level": "INFO",
    "min_line_length": 0,
    "name": "least_cost_transmission_1000MW"
}
```
If you would like to allow supply curve points to  connect to substations within the same state, add `"allow_connections_within_states": true` to the file.

Kickoff the execution using the following command:

    least-cost-xmission from-config -c least_cost_transmission_1000MW.json

You may need to run this command multiple times - once for each transmission line capacity.
As before the data will come split into multiple files (in this case 100, since we used 100 nodes). To merge the data, run a command similar to the one above:

    least-cost-xmission merge-output -of transmission_1000MW_128.csv -od /shared-projects/rev/transmission_tables/reinforced_transmission/least_cost_transmission least_cost_transmission_*_1000_128.csv

Finally, combine the spur line transmission costs and the reinforcement costs into a single transmission table:

    least-cost-xmission merge-reinforcement-costs -of transmission_reinforced_1000MW_128.csv -f /shared-projects/rev/transmission_tables/reinforced_transmission/least_cost_transmission/transmission_1000MW_128.csv -r /shared-projects/rev/transmission_tables/reinforced_transmission/reinforcement_costs/reinforcement_costs_400MW_230kV.csv

Again, you may need to run this command multiple times - once for each transmission line capacity.

The resulting tables can be passed directly to `reV`, which will automatically detect reinforcement costs and take them into account during the supply curve computation.

<br>


# Offshore Least Cost Paths

## Atlantic Offshore Wind Transmission (AOSWT) Workflow
General steps to run the AOSWT analysis:

1. Convert points-of-interconnection (POI) (grid connections on land) to transmission feature lines. Example notebook is at `reVX/examples/least_cost_paths/convert_points_of_interconnection_to_lines.ipynb`. The input CSV requires the following fields: 'POI Name', 'State', 'Voltage (kV)', 'Lat', 'Long'.
2. Create offshore friction and barrier (exclusion) layers and merge with CONUS costs and barrier layers. Example notebook is at `reVX/examples/least_cost_paths/combine_layers_and_add_to_h5.ipynb`.
3. Determine desired sc\_point_gids to process.
4. Select appropriate clipping radius. Unlike the CONUS analysis, which clips the cost raster by proximity to infinite sinks, the AOSWT uses a fixed search radius. 5000 is a good starting point. Note that memory usage increases with the square of radius.
5. Run analysis. See examples below.
6. Convert the output to GeoJSON (optional). See post processing below.


## Atlantic Offshore Wind Transmission (AOSWT) Examples
### Creating POI transmission features from points
The onshore point of interconnections (POIs) have typically been provided in a
CSV file. These must be converted to short lines in a GeoPackage to work with
the LCP code. Note that the POIs must also be connected to a transmission line.
The `convert_pois_to_lines()` function in `aoswt_utilities.py` will perform all
necessary operations to convert the CSV file to a properly configured
GeoPackage. An example notebook is in this repository at
`examples/least_cost_paths/convert_points_of_interconnection_to_lines.ipynb`.
Paths from POIs to the fake transmission line can be removed in post processing
using the `--drop TransLine` option with the `least-cost-xmission merge-output`
command.

### Build friction and barriers layer
An example Jupyter notebook for building the friction and barrier layers can be found in the `examples/least_cost_paths` directory of this repository.

### Locally run a AOSWT analysis for a single SC point, plot the results, and save to a GeoPackage
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

# Save to a GeoPackage
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
Using a config file is the preferred method of running an analysis. The below file processes a single SC point (sc_point_gid = 40139) on a debug node. Note that SLURM high quality of service on a standard node can be requested with `"feature": "--qos=high"`. This file also uses the optional `save_paths` and `radius` options to save the least coasts paths to a GeoPackage and force a cost raster clipping radius of 5000 pixels, versus determining the radius from the nearest sinks. Memory usage increases with the square of radius. Since this is an offshore analysis, the resolution SC point resolution is set to 118. The `simplify_geo` key is set to `100`. Be default, the saved paths will have vertices for each raster cell, resulting in very large output files. Using `simplify_geo` simplifies the geometry, greatly reduces output file sizes, and improves run times. Large number will result in less vertices and smaller files sizes.


The value for `allocation` should be set to the desired SLURM allocation. The `max_workers` key can be used to reduce the workers on each node if memory issues are encountered, but can typically be left out.

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
  "radius": 5000,
  "simplify_geo": 100
}
```

Assuming the above config file is saved as `config_aoswt.json` in the current directory, it can be kicked off with:

```
python -m reVX.least_cost_xmission.least_cost_xmission_cli from-config \
--config ./config_aoswt.json
```

### Post processing
Running an analysis on multiple nodes will result in multiple output files. These can be collected via several means. The below command will combine all output files into a single GeoPackage, assuming `save_paths` was enabled. If paths are not saved, the output will consist of multiple CSV files that must be merged manually.

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

Additionally, the results may be split into GeoJSONs by transmission feature connected to with the following. This will not create a combined GeoPackage file. The optional `--simplify-geo YYY` argument, where `YYY` is a number, can also be used if not set in the config file. Setting `simplify-geo` in the config file results in much faster run times than in post-processing.

```
python -m reVX.least_cost_xmission.least_cost_xmission_cli merge-output \
--drop TransLine \
--split-to-geojson --out-path ./out \
output_files_*.gpkg
```