# reVX Setbacks
The ``reVX`` setbacks module allows modeling of both local and national-level wind and solar siting ordinances as exclusions (for [``reV``](https://github.com/NREL/reV) and beyond).

This guide is supplemental to the [setbacks documentation](https://nrel.github.io/reVX/_cli/setbacks.html#setbacks) - please consult the latter for detailed explanations of all inputs.


<br>

## Computing setbacks for CONUS
### Ordinance database
The first step to computing setbacks is to obtain an up-to-date copy of the wind and/or solar local ordinance database
(csv file) from [``reVXOrdinances``](https://github.com/NREL/reVXOrdinances) (see the
[usage guide](https://github.com/NREL/reVXOrdinances/blob/main/docs/USAGE.md) for instructions).
You can skip this step if you only care about generic setbacks applied across the entire nation.

### Config file setup
Each reVX setbacks project should be run from a new directory. The execution logic assumes exactly one set of setbacks
configuration per directory. To begin, create a new directory for your project. The name should be descriptive since it
will show up in all output filenames. A good naming convention for wind, for example, is ``"wind_XXXhh_XXXrd"``
(i.e. "wind_116hh_163rd"). After you have created your directory, ``cd`` into it, and run the following
command (make sure your ``reVX`` environment has been activated):
```console
$ setbacks template-configs
```
This will create some template configuration files for you to use:
```console
$ ls
config_compute.json  config_merge.json  config_pipeline.json
```
You can change the config file type using the ``-t`` option for the ``template-configs`` command
(e.g. ``setbacks template-configs -t yaml``)

Open the ``config_compute.json`` template file. It should look something like this:
```json
{
    "execution_control": {
        "option": "local",
        "allocation": "[REQUIRED IF ON HPC]",
        "walltime": "[REQUIRED IF ON HPC]",
        "qos": "normal",
        "memory": null,
        "queue": null,
        "feature": null,
        "conda_env": null,
        "module": null,
        "sh_script": null,
        "max_workers": null
    },
    "log_directory": "./logs",
    "log_level": "INFO",
    "excl_fpath": "[REQUIRED]",
    "hub_height": null,
    "rotor_diameter": null,
    "base_setback_dist": null,
    "regulations_fpath": null,
    "weights_calculation_upscale_factor": null,
    "replace": false,
    "hsds": false,
    "out_layers": null,
    "feature_specs": null,
    "features": "[REQUIRED]",
    "generic_setback_multiplier": null
}
```
The template provides all possible input keys and clearly indicates which ones must be filled out. The rest of the inputs
are filled in with suitable default values, typically ``None`` (or ``null`` for JSON).

Begin by filling out the ``execution_control`` block. Typically, it is enough to simply update ``"option": "eagle"``
and just fill in the ``allocation`` and ``walltime`` parameters. The rest can be removed if unused.

The ``log_directory`` and ``log_level`` keys give you control over the location of output log files and their verbosity,
respectively.  Suitable options for ``log_level`` are ``DEBUG`` (most verbose), ``INFO`` (default),
``WARNING`` (only log warnings and errors), and ``ERROR`` (only log errors).

The next important parameter is ``excl_fpath``. This key must be a path that points to a template exclusions file
(path relative to the project directory are allowed). This file defines the raster grid - it must contain a county
FIPS layer called ``cnty_fips``. This layer is used to match local regulations in ``regulations_fpath`` to counties
on the grid.

If you are running setbacks for a particular wind turbine, fill out the ``hub_height`` and ``rotor_diameter`` inputs,
and **delete the ``base_setback_dist`` input**. ``reVX`` setbacks calculations do not allow ``base_setback_dist`` if the
``hub_height`` and ``rotor_diameter`` inputs pare provided, since it calculates ``base_setback_dist`` to be the max
tip-height of the turbine: ``base_setback_dist`` = ``hub_height`` + ``rotor_diameter`` / 2.

On the other hand, if you are calculating setbacks for solar (or some other technology), the ``hub_height`` and
``rotor_diameter`` inputs are meaningless. In this case, you **must** remove them and use the ``base_setback_dist``
input to specify the distance that will be scaled by the generic and local regulation multipliers.

The ``regulations_fpath`` should point to the [``reVX`` ordinance CSV file you generated above](#ordinance-database)
(path relative to the project directory are allowed). If you are not modeling any local regulations, you can leave this
input as ``None`` (``null`` in JSON).

Basic execution of ``reVX`` setbacks will not require ``weights_calculation_upscale_factor``, ``out_layers``, or
``feature_specs``, so we instead cover them in the [Advanced Topics section below](#advanced-topics). For now, these keys
are okay to remove from the configuration file entirely.

``replace`` and ``hsds`` can be left with their default values, unless you would like ``reVX`` to replace any existing
setbacks TIFFs in your project directory (``replace: true``) or your ``excl_fpath`` points to a file on AWS and you are
using HSDS to access it (``hsds: true``).

The ``features`` inputs points ``reVX`` to the feature data from which setbacks should be computed. The value for this
key should always be another dictionary. The keys in the new dictionary are the names of the setbacks you are computing
(see the keys of [``SETBACK_SPECS``](https://github.com/NREL/reVX/tree/main/reVX/setbacks/setbacks.py) for all possible
options - you may have to scroll down after clicking the link), and the values should point to the data on disk.
There are several ways to point to a data file (though all files must be ``GeoPackages``):
* If the features for a particular setback calculation are contained within a single file, just set the value to the path
  to the file (relative paths are allowed)
* If the features for a particular setback calculation are spread across several files (this is common practice to speed
  up execution; ``reVX`` will process each input file on a separate node in parallel), you have a few options:
  * You can provide a single unix-style wildcard path to specify the files (e.g. ``../my_data/*/*.gpkg``)
  * If you would like to run only particular files, or they are spread across multiple directories, or they do not fit
    within a wildcard pattern, you can specify a list of input paths. The paths in the list can be relative to the project
    directory, and they can also contain unix-style wildcards

Finally, you can specify a value for ``generic_setback_multiplier``. This is a multiplier value that will be applied to
``base_setback_dist`` to compute setback exclusions wherever a local ordinance is not given.

At this point, your config file may look something like this:
```json
{
    "execution_control": {
        "option": "eagle",
        "allocation": "revx",
        "walltime": 1
    },
    "log_level": "INFO",
    "excl_fpath": "/path/to/Exclusions.h5",
    "hub_height": 116,
    "rotor_diameter": 163,
    "regulations_fpath": "./wind_regulations.csv",
    "generic_setback_multiplier": 1.1,
    "features": {
        "rail": "/absolute/path/to/rail_data.gpkg",
        "transmission": "../../relative/path/to/transmission/data/*.gpkg",
        "road": [
            "../relative/path/to/first/road/data/file.gpkg",
            "/path/to/another/road/data/directory/multiple_files*.gpkg",
        ]

    }
}
```
This config would calculate setbacks using a ``base_setback_dist`` of 197.5m (max tip-height) onto a raster defined
vial the ``cnty_fips`` layer in ``"/path/to/Exclusions.h5"``. The local regulations would be pulled from
``"./wind_regulations.csv"``, and a generic multiplier of 1.1 would be applied to the max tip-height value everywhere
else. Three types of setbacks would be computed:
* Rail setbacks would be computed from all features in the file ``"/absolute/path/to/rail_data.gpkg"``
* Transmission line setbacks would be computed from all features in all files in the directory
  ``"../../relative/path/to/transmission/data"``
* Road setbacks would be computed from all features in the file ``"../relative/path/to/first/road/data/file.gpkg"``
  as well as the features in all files matching the pattern ``"multiple_files*.gpkg"`` in the directory
  ``"/path/to/another/road/data/directory"``

Once your setbacks run has been configured, you are ready to kick it off on the HPC.

### Execution
When you are ready to run setbacks computation, run the following command **from the project directory**:
```console
$ setbacks pipeline
```

This command runs the first step in the setbacks pipeline (which was configured for you automatically in
``config_pipeline.json``). This command can be called repeatedly to execute the next step of the pipeline once
the current step is done running. It can also be used to re-run a failed or partially-successful step.

Although it is not recommended, you can also run
```console
$ setbacks pipeline --background
```
to execute all steps in the pipeline, one after another, without any other user intervention. Be aware, however, that
this execution will be interrupted if any job fails for any reason, so you may still have to submit this call multiple
times.

If your first step executes successfully, you should see one or more output TIFF files in your project directory
(specifically, one output file per input file):
```console
$ ls *.tif

setbacks_rail_wind_116m_163m_j00.tif
setbacks_transmission_wind_116m_163m_j01.tif
setbacks_transmission_wind_116m_163m_j02.tif
setbacks_transmission_wind_116m_163m_j03.tif
...
setbacks_transmission_wind_116m_163m_j32.tif
setbacks_road_wind_116m_163m_j33.tif
setbacks_road_wind_116m_163m_j34.tif
setbacks_road_wind_116m_163m_j35.tif
...
```

Typically the next step is to merge the setbacks computed separately across all the input files into a single file per
setback type.

### Merging
The next (and final) step in the auto-generated pipeline will merge all the setback files for a particular setback type
into a single TIFF file (the underlying assumption here is that the input files for each feature type do not overlap
spatially). If you do not wish to merge the files, simply skip this step.

Before submitting the merge step to the HPC, open the ``config_merge.json`` file and update the ``execution_control``
block like you did before. The rest of the default inputs can be left as-is. Once the ``execution_control`` has been
updated, run the following command **from the project directory**:
```console
$ setbacks pipeline
```

This will submit the "merge" step. Once this step has finished running, you should see a single TIFF file per
setback type in your directory (along with a ``chunk_files`` folder containing the individual TIFF files from the
previous step):
```console
$ ls
chunk_files
...
setbacks_rail_wind_116m_163m.tif
setbacks_transmission_wind_116m_163m.tif
setbacks_road_wind_116m_163m.tif
...
```

Congratulations, you have now computed setbacks using ``reVX``!

If you need to move the output data into an HDF5 file
to be as ``reV`` exclusion layers, you can use the
[``reVX exclusions layers-to-h5`` command](https://nrel.github.io/reVX/_cli/reVX.html#revx-exclusions-layers-to-h5).

<br>

## Advanced Topics
In this section, we explore some more complex use-patterns that ``reVX`` supports for setbacks exclusion calculations.

### Partial Setbacks
The size of some features you may want to calculate setbacks for may be on the order of (or even smaller!)
than your exclusion grid size (e.g. parcels). In these cases, it's useful to calculate *partial* setback
exclusions, where pixels in your grid are not simply a binary flag but rather partial exclusion values.
``reVX`` supports this type of calculation - all you have to do is specify the ``weights_calculation_upscale_factor`` in
your ``config_compute.json`` to be a value larger than 1. Under the hood, this upscales your exclusion grid by that factor during the setback calculation and uses the higher-resolution grid to calculate the partial area that should be excluded.

> **Warning** <br>
If you set ``weights_calculation_upscale_factor`` > 1 in your ``config_compute.json``, your output
data will now be an *inclusion* mask (as opposed to an *exclusion* layer), where each pixel will contain a float indicating
the fractional inclusion weight. In other words, a value of 1 represents 100% *inclusion*, a value of 0.75 represents a 75%
*inclusion*, and a value of 0 represents 0%  *inclusion*, or full exclusion. This is **the opposite** of normal setback
exclusions outputs, where the output values are bools with ``1 == exclusion`` and ``0 == inclusion``. The reason for this
discrepancy is for direct coupling with ``reV``, which expects all partial exclusions to be input as an *inclusion* mask.

### Writing directly to HDF5 files
``reVX`` supports writing the output setback data directly to the ``excl_fpath`` exclusions h5 file in addition to
an output TIFF file. This is rarely useful for setbacks, since the input features are often broken out over may files
and thus the output needs to be merged before writing to an exclusion layer. Nevertheless, if your features come in a
single input file (or you really like having hundreds of layers in your h5 files), you can request to have the output data
stored directly in the ``excl_fpath`` file by including the ``out_layers`` key in your config file:
```
"out_layers": {
    "rail_data.gpkg": "rail_setbacks_116hh_163rd",
    "transmission_az.gpkg": "transmission_setbacks_116hh_167rd_az",
    "transmission_il.gpkg": "transmission_setbacks_116hh_167rd_il",
    ...
}
```
Note that you are mapping the input data files directly to the layer where the output setback data should be stored.

### Feature-specific generic multipliers
Often, you may wish to model a different generic multiplier for each type of setback feature. Instead of setting up many
different configuration files with one feature type each, ``reVX`` lets you specify feature-specific generic multipliers
in a separate config file. For example, suppose you create a file ``generic_multipliers.json`` with the following contents:
```json
{
    "road": 1.5,
    "parcel": 1.1,
    "structure": 5
}
```
Then, in your ``config_compute.json`` config file, instead of specifying a single value for ``generic_setback_multiplier``,
you can simply point to the new config:
```json
"generic_setback_multiplier": "./generic_multipliers.json"
```

With this configuration, reVX will use a multiplier of 1.5 for all generic road setbacks, a multiplier of 1.1 for all
generic parcel setbacks, and a multiplier of 5 for all generic structure setbacks. Note that you **must** provide a
multiplier for each feature type you specify in the ``features`` input of your ``config_compute.json`` config file.

### Custom setbacks computations
Sometimes a user may want to compute setbacks from a feature type that is not explicitly supported in ``reVX``.
A historical example of this are setbacks from oil and natural gas pipelines. As of April 14, 2023, ``reVX`` still
does not provide explicit support for calculating setbacks from pipelines, even though setbacks from roads and transmission lines are conceptually and computationally similar to pipeline setbacks.

In order to compute such setbacks, users can create their own "feature types" using the ``feature_specs`` input.
This input must be a dictionary where keys are the names of the new setback types, and the values are also dictionaries
containing keyword-value pairs for the [``setbacks_calculator``](https://nrel.github.io/reVX/_autosummary/reVX.setbacks.setbacks.setbacks_calculator.html#reVX.setbacks.setbacks.setbacks_calculator) function. For example, the input:
```json
"feature_specs": {
    "oil_and_gas": {
        "feature_type": "Oil And Gas Pipelines",
        "buffer_type":"default",
        "feature_filter_type":"clip",
        "feature_subtypes_to_exclude": null,
        "num_features_per_worker": 10000
    }
}
```
would define a new feature type ``"oil_and_gas"`` that would be identified in the regulations CSV file as
``"Oil And Gas Pipelines"`` under the ``"Feature Type"`` column. Only the ``feature_type`` key is required in the
inner-most dictionary (default values are provided by the function for all other keys). In this case, we updated
the ``feature_filter_type`` to ``"clip"`` instead of centroid, because we want pipelines ot be clipped to the county
for which setbacks are being computed (as opposed to requiring the centroid to be within the county - the centroid
may be quite far away depending on the shape of the pipeline). For more details on the input keys for each new
feature type, please see the documentation for the [``setbacks_calculator``](https://nrel.github.io/reVX/_autosummary/reVX.setbacks.setbacks.setbacks_calculator.html#reVX.setbacks.setbacks.setbacks_calculator) function.

After adding the above input to the ``config_compute.json`` config file, you can use ``"oil_and_gas"`` just like any of
the "standard" feature types. In particular, you can specify this feature in the ``features`` input:
```
"features": {
    ...
    "oil_and_gas": "/path/to/oil_gas/pipelines.gpkg",
    ...
}
```
and the ``generic_setback_multiplier`` config input:
```json
{
    "road": 1.5,
    "parcel": 1.1,
    "structure": 5,
    "oil_and_gas": 3
}
```

The ``feature_specs`` input can also be used to calculate the same type of setback for two separate input datasets
simultaneously:
```json
"feature_specs": {
    "water-nwi": {
        "feature_type": "water",
        "buffer_type": "default",
        "feature_filter_type": "clip",
        "num_features_per_worker": 1000,
    },
    "water-nhd": {
        "feature_type": "water",
        "buffer_type": "default",
        "feature_filter_type": "clip",
        "num_features_per_worker": 1000,
    }
}
```
Of course, you would have to point ``"water-nwi"`` and ``"water-nhd"`` to separate input datasets using the
``features`` input.


### Batched execution
Although ``reVX`` provides a lot of flexibility when it comes to calculating different setbacks for a single turbine
(or other technology specification), the setup can still become cumbersome when working with even a handful of technology/
siting scenarios. To facilitate setup in these cases, users are encouraged to use the ``batch`` functionality (provided by
the underlying GAPs framework). There are several ways to set up batched setbacks runs. Here will will focus on the
CSV input method.

First, we need to generate a csv config file that tells ``batch`` how we want to parameterize our runs.
Let's suppose that we want to compute setbacks for five different turbines, each with three different siting (setback)
scenarios. Here is a short python script to generate the CSV config file:

```python
# make_batch_csv.py
import json
from itertools import product
import pandas as pd

# Define generic mults for the three different siting scenarios (open has no mults)
reference_access_generic_mults = {"rail": 1.5, "road": 2.1, "structure": 3.2, "transmission": 1.7, "water": 2}
limited_access_generic_mults = {"rail": 3, "road": 4, "structure": 5, "transmission": 2, "water": 3}

# Define generic mults config filepaths
ra_fp = "./reference_access_generic_mults.json"
la_fp = "./limited_access_generic_mults.json"

# Write generic mults to files
with open(ra_fp, "w") as fh:
    json.dump(reference_access_generic_mults, fh)
with open(la_fp, "w") as fh:
    json.dump(limited_access_generic_mults, fh)

# Define our parametrizations
mults = {"open": None, "reference": ra_fp, "limited": la_fp}
turbines = [(120, 90), (150, 100), (170, 120), (190, 145), (190, 160)]  # (rd, hh)

# Generate our parametrizations
rows = []
for (rd, hh), scenario in product(turbines, mults):
    tag = f"{scenario}_{hh}hh_{rd}rd"  # tag is used as the directory name for each run
    rows.append([tag, rd, hh, mults[scenario]])

# Generate a pandas DataFrame from our parametrizations
# `batch` will update the "rotor_diameter", "hub_height", "generic_setback_multiplier" inputs in our
# `config_compute.json` file to the required value for the parametric run
batch = pd.DataFrame(rows, columns=["set_tag", "rotor_diameter", "hub_height", "generic_setback_multiplier"])

# Next two columns required by the `batch` command.
batch["pipeline_config"] = "./config_pipeline.json"  # point to pipeline config file
batch["files"] = "['./config_compute.json']"  # Python list of all the files batch should update, wrapped in quotes
batch.to_csv("config_batch.csv", index=False)
```

Running this script, we get the following table as output:


|set_tag|rotor_diameter|hub_height|generic_setback_multiplier|pipeline_config|files|
|-------|--------------|----------|--------------------------|---------------|-----|
|open_90hh_120rd|120|90|N/A|./config_pipeline.json|['./config_compute.json']
|reference_90hh_120rd|120|90|./reference_access_generic_mults.json|./config_pipeline.json|['./config_compute.json']
|limited_90hh_120rd|120|90|./limited_access_generic_mults.json|./config_pipeline.json|['./config_compute.json']
|open_100hh_150rd|150|100|N/A|./config_pipeline.json|['./config_compute.json']
|reference_100hh_150rd|150|100|./reference_access_generic_mults.json|./config_pipeline.json|['./config_compute.json']
|limited_100hh_150rd|150|100|./limited_access_generic_mults.json|./config_pipeline.json|['./config_compute.json']
|open_120hh_170rd|170|120|N/A|./config_pipeline.json|['./config_compute.json']
|reference_120hh_170rd|170|120|./reference_access_generic_mults.json|./config_pipeline.json|['./config_compute.json']
|limited_120hh_170rd|170|120|./limited_access_generic_mults.json|./config_pipeline.json|['./config_compute.json']
|open_145hh_190rd|190|145|N/A|./config_pipeline.json|['./config_compute.json']
|reference_145hh_190rd|190|145|./reference_access_generic_mults.json|./config_pipeline.json|['./config_compute.json']
|limited_145hh_190rd|190|145|./limited_access_generic_mults.json|./config_pipeline.json|['./config_compute.json']
|open_160hh_190rd|190|160|N/A|./config_pipeline.json|['./config_compute.json']
|reference_160hh_190rd|190|160|./reference_access_generic_mults.json|./config_pipeline.json|['./config_compute.json']
|limited_160hh_190rd|190|160|./limited_access_generic_mults.json|./config_pipeline.json|['./config_compute.json']

This table tells ``batch`` to create 15 different run folders (one for each ``set_tag``) and update the
``rotor_diameter``, ``hub_height``, and ``generic_setback_multiplier`` input keys appropriately for each run. The two extra
things ``batch`` needs to know are the path to the pipeline file to run for each folder (``pipeline_config`` column) and
the config files that ``batch`` should look in for our input keys (``rotor_diameter``, ``hub_height``, and
``generic_setback_multiplier``). The latter is given in the ``files`` column (note that the syntax to fill this
value using the python script above resembles a string representation of a Python list).

Once this config CSV file is generated, generate the rest of the config files and fill them out, as outlined above.
Your directory should look like this:
```console
$ ls
config_batch.csv
config_compute.json
config_merge.json
config_pipeline.json
limited_access_generic_mults.json
make_batch_csv.py
reference_access_generic_mults.json
```

At this point, you can run
```console
$ setbacks batch -c config_batch.csv
```

This command will create 15 subdirectories (with ``set_tag`` for names), copy over all relevant config files, and
kickoff the pipeline job in each directory:

```console
$ ls
...
limited_100hh_150rd
limited_120hh_170rd
limited_145hh_190rd
limited_160hh_190rd
limited_90hh_120rd
...
open_100hh_150rd
open_120hh_170rd
open_145hh_190rd
open_160hh_190rd
open_90hh_120rd
...
reference_100hh_150rd
reference_120hh_170rd
reference_145hh_190rd
reference_160hh_190rd
reference_90hh_120rd
...
```

After the first step of the pipeline completes for all of the sub-directories, you will have to run
```console
$ setbacks batch -c config_batch.csv
```
again to kickoff the "merge" step. Once the "merge" step completes, you have computed setbacks for 15 different
turbine/siting combinations!


### Job Status
You can check the status of a project directory by running
```console
$ setbacks status
```
This command will print a table of submitted/running/completed jobs for a particular project directory.
The jobs will be identified using a tag. In order to see the input file being processed by each job run the
following command:
```console
$ setbacks status -i node_file_path
```