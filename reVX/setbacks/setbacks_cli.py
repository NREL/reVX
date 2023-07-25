# -*- coding: utf-8 -*-
"""
Setbacks CLI
"""
import os
import glob
import shutil
import logging
from pathlib import Path
from warnings import warn

from gaps.config import load_config
from gaps.cli import CLICommandFromFunction, make_cli
from gaps.cli.preprocessing import preprocess_collect_config
from reVX.setbacks import SETBACKS
from reVX.setbacks.regulations import (validate_setback_regulations_input,
                                       select_setback_regulations)
from reVX.setbacks.setbacks_converter import parse_setbacks
from reVX.handlers.geotiff import Geotiff
from reVX.utilities import ExclusionsConverter
from reVX import __version__
from reVX.setbacks.setbacks import SETBACK_SPECS, setbacks_calculator


logger = logging.getLogger(__name__)


def _validate_multipliers(features, generic_setback_multiplier=None):
    """Load, format, and validate generic multipliers. """
    if isinstance(generic_setback_multiplier, str):
        logger.debug("Loading multipliers from {!r}"
                     .format(generic_setback_multiplier))
        generic_setback_multiplier = load_config(generic_setback_multiplier)

    try:
        missing = {feat for feat in features
                   if feat not in generic_setback_multiplier}
    except TypeError:
        return {feat: generic_setback_multiplier for feat in features}

    if missing:
        msg = ("The following features are missing from the "
               "`generic_cost_multiplier` input: {}. Please make sure every "
               "key in the `features` input has a matching key in the "
               "`generic_cost_multiplier` dictionary"
               .format(missing))
        logger.error(msg)
        raise KeyError(msg)
    return {feat: generic_setback_multiplier[feat] for feat in features}


def preprocess_setbacks_config(config, features,
                               generic_setback_multiplier=None):
    """Pre-process setbacks computation config.

    Parameters
    ----------
    config : dict
        Setbacks compute config. This config will be updated to include
        a the ``node_feature_type``, ``node_file_path``, and
        ``node_multiplier`` keys based on user input.
    features : dict
        Dictionary specifying which features/data to process. The keys
        of this dictionary must be the a key from the
        :attr:`~reVX.setbacks.setbacks.SETBACK_SPECS` dictionary or the
        ``feature_specs`` input dictionary specifying the feature type
        to run setbacks for. The value of each key must be a path or a
        list of paths to calculate that particular setback for.
        The path(s) can contain unix-style file-pattern matching syntax
        to point to multiple files. The paths may be specified relative
        to the config file. For example::

            features: {
                "parcel": "../relative/path/to/parcel_colorado.gpkg",
                "road": [
                    "/full/path/to/road/data/*.gpkg",
                    "../../relative/path/to/data_i[l,n].gpkg",
                ]
            }

        With this input, parcel setbacks would be computed for the
        data in ``../relative/path/to/parcel_colorado.gpkg``, and road
        setbacks would be calculated for *all* GeoPackage data files in
        ``/full/path/to/road/data/`` and for the files
        ``../../relative/path/to/data_il.gpkg`` and
        ``../../relative/path/to/data_in.gpkg``.
    generic_setback_multiplier : int | float | str, optional
        Optional setback multiplier to use where local regulations are
        not supplied. This multiplier will be applied to the
        ``base_setback_dist`` (or the turbine tip-height) to calculate
        the setback. If supplied along with ``regulations_fpath``, this
        input will be used to apply a setback to all counties not listed
        in the regulations file. This input can also be a path to a
        config file containing feature types as keys and
        feature-specific generic multipliers as values. For example::

            {
                "parcel": 1.1,
                "road": 2,
                "structure": 3.5
            }

        If specified this way, every key in the ``features`` inputs
        must also be given in the generic multipliers config. If
        ``None``, no  generic setback computation is performed.
        By default, ``None``.


    Returns
    -------
    config : dict
        Updated setbacks computation config dictionary.

    Raises
    ------
    ValueError
        If ``features`` is not a dictionary, or any key in ``features``
        is not a valid key of :attr:`SETBACKS`.
    FileNotFoundError
        If no input GeoPackage files are found on disk.
    """
    if not isinstance(features, dict):
        raise ValueError("`features` key must be a dictionary, got {}"
                         .format(features))
    feature_specs = config.get("feature_specs", {})
    combos_to_run = []
    multipliers = _validate_multipliers(features, generic_setback_multiplier)
    for feature_type, features_fpath in features.items():
        if feature_type not in set(SETBACKS) | set(feature_specs):
            msg = ("feature_type must be one of: {}; got {!r}"
                   .format(set(SETBACKS.keys()), feature_type))
            raise ValueError(msg)

        if isinstance(features_fpath, str):
            features_fpath = [features_fpath]

        for path in features_fpath:
            glob_path = Path(path).expanduser().resolve()
            if glob_path.is_dir():
                glob_path = glob_path / '*'

            need_to_run = [(feature_type, str(f), multipliers[feature_type])
                           for f in glob_path.parent.glob(glob_path.name)
                           if f.name.endswith("gpkg")]

            if not need_to_run:
                msg = ("Found no unprocessed GeoPackage files matching the "
                       "input {!r}!".format(path))
                logger.warning(msg)
                warn(msg)
                continue

            combos_to_run += need_to_run

    if not combos_to_run:
        msg = 'No unprocessed GeoPackage files found!'
        logger.error(msg)
        raise FileNotFoundError(msg)

    feature_type, file_path, multiplier = zip(*sorted(combos_to_run))
    config["node_feature_type"] = feature_type
    config["node_file_path"] = file_path
    config["node_multiplier"] = multiplier
    validate_setback_regulations_input(config.get("base_setback_dist"),
                                       config.get("hub_height"),
                                       config.get("rotor_diameter"))
    _update_setbacks_calculators(feature_specs)  # test for errors
    return config


def preprocess_merge_config(config, project_dir, command_name,
                            merge_file_pattern="PIPELINE"):
    """Pre-process merge config.

    Parameters
    ----------
    config : dict
        Collection config. This config will be updated to include the
        keys ``node_out_path`` and ``node_pattern`` representing the
        output file path and the input file pattern, respectively.
    project_dir : path-like
        Path to project directory. This path is used to resolve the
        output filepath.
    command_name : str
        Name of the command being run. This is used to parse the
        pipeline status for output files if
        ``"merge_file_pattern": "PIPELINE"`` in the input `config`.
    merge_file_pattern : str | list | dict, optional
        Unix-style ``/filepath/pattern*.h5`` representing the files to
        be merged into a single output GeoTIFF file. If no output file
        path is specified (i.e. this input is a single pattern or a list
        of patterns), the output file path will be inferred from the
        pattern itself (specifically, the wildcard will be removed
        and the result will be the output file path). If a list of
        patterns is provided, each pattern will be merged into a
        separate output file. To specify the name of the output file(s),
        set this input to a dictionary whose keys are paths to the
        output file (relative paths are allowed) and the values are
        patterns representing the input files that should be merged into
        the output TIFF. If running a merge job as part of a pipeline,
        this input can be set to ``"PIPELINE"``, which will parse the
        output of the previous step (``compute``) and generate the input
        file pattern and output file name automatically.
        By default, ``"PIPELINE"``.

    Returns
    -------
    config : dict
        Updated merge config dictionary.
    """
    config = preprocess_collect_config(config, project_dir, command_name,
                                       collect_pattern=merge_file_pattern)
    config["node_out_path"] = config.pop("_out_path", None)
    config["node_pattern"] = config.pop("_pattern", None)
    return config


def _update_setbacks_calculators(feature_specs=None):
    """Update `SETBACKS` to include new specs """

    for feature_name, feature_spec in (feature_specs or {}).items():
        spec = SETBACK_SPECS.get(feature_name, {})
        spec.update(feature_spec)
        SETBACKS[feature_name] = setbacks_calculator(**spec)


def compute_setbacks(excl_fpath, node_feature_type, node_file_path,
                     node_multiplier, out_dir, tag, hub_height=None,
                     rotor_diameter=None, base_setback_dist=None,
                     regulations_fpath=None,
                     weights_calculation_upscale_factor=None,
                     replace=False, hsds=False, out_layers=None,
                     feature_specs=None, max_workers=None):
    """Compute Setbacks.

    Setbacks can be computed for a specific turbine (hub height and
    rotor diameter) or more generally using a base setback distance.

    Setbacks can be computed either locally (on a per-county basis with
    given distances/multipliers) or everywhere under a generic setback
    multiplier assumption applied to either the turbine tip-height or
    the base setback distance. These two methods can also be applied
    simultaneously - local setbacks are computed where given (via a the
    regulation file input) and a generic multiplier applied to the
    turbine tip-height or the base setback distance everywhere else.

    Partial inclusions can be computed instead of boolean exclusions,
    both of which can be fed directly into ``reV``.

    Parameters
    ----------
    excl_fpath : str
        Path to HDF5 file containing the county FIPS layer (should be
        called ``cnty_fips``) used to match local regulations in
        ``regulations_fpath`` to counties on the grid. No data will be
        written to this file unless explicitly requested via the
        ``out_layers`` input.
    node_feature_type : str
        Name of the feature type being run. Must be a key of the
        :attr:`SETBACKS` dictionary.
    node_file_path : str
        Path to input feature file. This file MUST be a GeoPackage (and
        have the ".gpkg" extension).
    node_multiplier : int | float | str | None, optional
        A setback multiplier to use if regulations are not supplied.
        This multiplier will be applied to the ``base_setback_dist``
        to calculate the setback. If supplied along with
        ``regulations_fpath``, this input will be used to apply a
        setback to all counties not listed in the regulations file.
    out_dir : str
        Path to output directory where output file should be written.
    tag : str
        Tag to add to each output file to make it unique (i.e. not clash
        with output files from other nodes).
    hub_height : int | float, optional
        Turbine hub height (m), used along with rotor diameter to
        compute the blade tip-height which is used as the base setback
        distance for generic/local regulations. If this input is
        specified, ``rotor_diameter`` must also be given, and
        ``base_setback_dist`` *must be set to None*, otherwise an
        error in thrown. The base setback distance is scaled by
        generic/local multipliers (provided either via the
        ``regulations_fpath`` csv, or the ``generic_setback_multiplier``
        input, or both) before setbacks are computed.
        By default, ``None``.
    rotor_diameter : int | float, optional
        Turbine rotor diameter (m), used along with hub height to
        compute the blade tip-height, which is used as the base setback
        distance for generic/local regulations. If this input is
        specified, ``hub_height`` must also be given, and
        ``base_setback_dist`` *must be set to None*, otherwise an
        error in thrown. The base setback distance is scaled by
        generic/local multipliers (provided either via the
        ``regulations_fpath`` csv, or the ``generic_setback_multiplier``
        input, or both) before setbacks are computed.
        By default, ``None``.
    base_setback_dist : int | float, optional
        Base setback distance (m). This value is used as the base
        setback distance for generic/local regulations. If this input is
        specified, both ``hub_height``and ``rotor_diameter`` *must be
        set to None*, otherwise an error in thrown. The base setback
        distance is scaled by generic/local multipliers (provided either
        via the ``regulations_fpath`` csv, or the
        ``generic_setback_multiplier`` input, or both) before setbacks
        are computed. By default, ``None``.
    regulations_fpath : str, optional
        Path to regulations ``.csv`` or ``.gpkg`` file. At a minimum,
        this file must contain the following columns: ``Feature Type``,
        which contains labels for the type of setback that each row
        represents, ``Feature Subtype``, which contains labels for
        feature subtypes, ``Value Type``, which specifies wether the
        value is a multiplier or static height, ``Value``, which
        specifies the numeric value of the setback or multiplier, and
        ``FIPS``, which specifies a unique 5-digit code for each county
        (this can be an integer - no leading zeros required). See
        :obj:`~reVX.setbacks.regulations.SetbackRegulations` (if using
        only ``base_setback_dist`` input) or
        :obj:`~reVX.setbacks.regulations.WindSetbackRegulations` (if
        using ``hub_height`` + ``rotor_diameter`` input) for more info.
        This option overrides the ``generic_setback_multiplier`` input,
        but only for counties that are listed in the input CSV file.
        This means both ``regulations_fpath`` and
        ``generic_setback_multiplier`` can be specified
        simultaneously in order to compute setbacks driven by local
        ordinance where given + a generic multiplier applied everywhere
        else. By default, ``None``, which does not compute any
        local setbacks.
    weights_calculation_upscale_factor : int, optional
        Optional input to specify *partial* setback calculations.
        If this value is an int > 1, the output will be a layer with
        **inclusion** weight values (floats ranging from 0 to 1). Note
        that this is backwards w.r.t the typical output of exclusion
        integer values (1 for excluded, 0 otherwise). Values <= 1 will
        still return a standard exclusion mask. For example, a cell that
        was previously excluded with a boolean mask (value of 1) may
        instead be converted to an inclusion weight value of 0.75,
        meaning that 75% of the area corresponding to that point should
        be included (i.e. the exclusion feature only intersected a small
        portion - 25% - of the cell). This percentage inclusion value is
        calculated by upscaling the output array using this input value,
        rasterizing the exclusion features onto it, and counting the
        number of resulting sub-cells excluded by the feature. For
        example, setting the value to ``3`` would split each output cell
        into nine sub-cells - 3 divisions in each dimension. After the
        feature is rasterized on this high-resolution sub-grid, the area
        of the non-excluded sub-cells is totaled and divided by the area
        of the original cell to obtain the final inclusion percentage.
        Therefore, a larger upscale factor results in more accurate
        percentage values. If ``None`` (or a value <= 1), this process
        is skipped and the output is a boolean exclusion mask.
        By default ``None``.
    replace : bool, optional
        Flag to replace the output GeoTIFF if it already exists.
        By default, ``False``.
    hsds : bool, optional
        Boolean flag to use ``h5pyd`` to handle HDF5 "files" hosted on
        AWS behind HSDS. By default, ``False``.
    out_layers : dict, optional
        Dictionary mapping the input feature file names (with extension)
        to names of layers under which exclusions should be saved in the
        ``excl_fpath`` HDF5 file. If ``None`` or empty dictionary,
        no layers are saved to the HDF5 file. By default, ``None``.
    feature_specs : dict, optional
        Optional dictionary specifying new feature setback calculators
        or updates to existing ones. The keys of this dictionary should
        be names of the features for which a specification is being
        provided. If the name is already a key in
        :attr:`~reVX.setbacks.setbacks.SETBACK_SPECS`, the corresponding
        specifications wil be updated for that feature. Otherwise, the
        name will represent a new feature type, which can be used as a
        key in the ``features`` input. The values of the feature-type
        keys should be dictionaries, where the keys are parameters of
        the :func:`~reVX.setbacks.setbacks.setbacks_calculator`
        function. Required parameters in that function are required keys
        of these dictionaries. Values should be the updated value.
        For example, the input
        ::

            feature_specs: {
                "water": {
                    "num_features_per_worker": 500
                },
                "oil_and_gas_pipelines": {
                    "feature_type": "oil and gas",
                    "feature_filter_type": "clip"
                }
            }

        would update the existing ``"water"`` setbacks calculator to
        compute 500 features per worker at a time and create a new
        ``"oil_and_gas_pipelines"`` feature that looks for the string
        ``"oil and gas"`` in the regulations file and clips the feature
        to a county before calculating a setback. Note that even though
        ``"oil_and_gas_pipelines"`` is not a default feature supported
        by ``reVX``, you can now use it in the ``features`` input.
        This can also be helpful if you need to compute the same type of
        setback for multiple different input datasets. For example, the
        input
        ::

            feature_specs: {
                "water-nwi": {
                    "feature_type": "water",
                    "buffer_type": "default",
                    "feature_filter_type": "clip",
                    "num_features_per_worker": 700,
                },
                "water-nhd": {
                    "feature_type": "water",
                    "buffer_type": "default",
                    "feature_filter_type": "clip",
                    "num_features_per_worker": 10_000,
                }
            }

        would allow you to set up your ``features`` input like so::

            features: {
                "water-nwi": "/path/to/nwi/*.gpkg",
                "water-nhd": "/path/to/nhd/*.gpkg",
            }

        By default, ``None``, which does not add any new setback
        calculators (the default ones defined in
        :attr:`~reVX.setbacks.setbacks.SETBACK_SPECS` are still
        available).
    max_workers : int, optional
        Number of workers to use for setback exclusion computation. If
        this value is 1, the computation runs in serial. If this value
        is > 1, the computation runs in parallel with that many workers.
        If ``None``, the computation runs in parallel on all available
        cores. By default, ``None``.

    Returns
    -------
    str
        Path to output GeoTIFF file containing setback exclusion data.
    """

    _update_setbacks_calculators(feature_specs)
    logger.info('Computing setbacks from {} in {}'
                .format(node_feature_type, node_file_path))
    logger.debug('Setbacks to be computed with:\n'
                 '- base_setback_dist = {}\n'
                 '- hub_height = {}\n'
                 '- rotor_diameter = {}\n'
                 '- regulations_fpath = {}\n'
                 '- generic_setback_multiplier = {}\n'
                 '- using max_workers = {}\n'
                 '- replace layer if needed = {}\n'
                 '- weights calculation upscale factor = {}\n'
                 '- out_layers = {}\n'
                 .format(base_setback_dist, hub_height, rotor_diameter,
                         regulations_fpath, node_multiplier, max_workers,
                         replace,
                         weights_calculation_upscale_factor, out_layers))

    regulations = select_setback_regulations(base_setback_dist, hub_height,
                                             rotor_diameter, regulations_fpath,
                                             node_multiplier)
    setbacks_class = SETBACKS[node_feature_type]
    wcuf = weights_calculation_upscale_factor
    fn = ("setbacks_{}_{}{}.tif"
          .format(node_feature_type, os.path.basename(out_dir), tag))
    out_fn = os.path.join(out_dir, fn)
    setbacks_class.run(excl_fpath, node_file_path, out_fn, regulations,
                       weights_calculation_upscale_factor=wcuf,
                       max_workers=max_workers, replace=replace, hsds=hsds,
                       out_layers=out_layers)
    logger.info('Setbacks computed and written to {}'.format(out_fn))
    return out_fn


def merge_setbacks(node_out_path, node_pattern, are_partial_inclusions=None,
                   purge_chunks=False):
    """Combine many input setback GeoTIFFs into a single layer.

    Parameters
    ----------
    node_out_path : str
        Path to output GeoTIFF file.
    node_pattern : str
        Input GeoTIFF file pattern.
    are_partial_inclusions : bool, optional
        Flag indicating wether the inputs are partial inclusion values
        or boolean exclusions. If ``None``, will try to infer
        automatically from the input file's GeoTIFF profile
        (``dtype != uint8``). By default, ``None``.
    purge_chunks : bool, optional
        Flag indicating wether individual "chunk" files should be
        deleted after a successful merge (``True``), or if they should
        be stored in a "chunk_files" directory (``False``).
        By default, ``False``.

    Raises
    ------
    FileNotFoundError
        If no input files found on disk matching the input pattern.
    RuntimeError
        If the ``are_partial_inclusions`` cannot be inferred (GeoTIFF
        profile does not have "dtype" field).
    """
    out_file = Path(node_out_path).resolve()

    logger.info("Merging TIFF files in {!r} and writing to {!r}"
                .format(out_file.parent.as_posix(), out_file))
    input_setback_files = list(glob.glob(node_pattern))
    if not input_setback_files:
        msg = ("Did not find any files matching pattern {!r} in directory {!r}"
               .format(node_pattern, out_file.parent.as_posix()))
        logger.error(msg)
        raise FileNotFoundError(msg)

    with Geotiff(input_setback_files[0]) as tif:
        profile = tif.profile

    if are_partial_inclusions is None:
        data_dtype = profile.get("dtype")
        if data_dtype is None:
            msg = ("Cannot infer whether data are partial inclusions because "
                   "data dtype is not set in the GeoTIFF profile. Please "
                   "explicitly specify the `are_partial_inclusions` input!")
            logger.error(msg)
            raise RuntimeError(msg)

        are_partial_inclusions = data_dtype.casefold() != "uint8"

    setbacks = parse_setbacks(input_setback_files,
                              is_inclusion_layer=are_partial_inclusions)

    logger.info("Writing data to {!r}".format(out_file.as_posix()))
    ExclusionsConverter.write_geotiff(out_file.as_posix(), profile, setbacks)

    if purge_chunks:
        for fpath in input_setback_files:
            os.remove(fpath)
    else:
        chunk_dir = out_file.parent / "chunk_files"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for fpath in input_setback_files:
            shutil.move(fpath, chunk_dir / os.path.basename(fpath))

        logger.info("Moved chunk files from {} to sub_dir: {}"
                    .format(out_file.parent, chunk_dir))


PRIVATE_COMPUTE_KEYS = ("node_feature_type", "node_file_path",
                        "node_multiplier")
PRIVATE_MERGE_KEYS = ("node_out_path", "node_pattern")
commands = [
    CLICommandFromFunction(
        function=compute_setbacks, name="compute",
        split_keys=[PRIVATE_COMPUTE_KEYS],
        config_preprocessor=preprocess_setbacks_config,
        skip_doc_params=PRIVATE_COMPUTE_KEYS,
    ),
    CLICommandFromFunction(
        function=merge_setbacks, name="merge",
        split_keys=[PRIVATE_MERGE_KEYS],
        config_preprocessor=preprocess_merge_config,
        skip_doc_params=PRIVATE_MERGE_KEYS,
    ),
]

cli = make_cli(commands)


if __name__ == '__main__':
    try:
        cli(obj={})
    except Exception:
        logger.exception('Error running Setbacks CLI')
        raise
