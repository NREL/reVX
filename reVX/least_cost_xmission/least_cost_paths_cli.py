# -*- coding: utf-8 -*-
# pylint: disable=all
"""
Least Cost Xmission Command Line Interface
"""
import time
import click
import logging
import os
import json
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj.crs import CRS
from shapely.geometry import Point

from rex.utilities.loggers import init_logger, init_mult, create_dirs
from rex.utilities.cli_dtypes import STR, INT
from rex.utilities.hpc import SLURM
from rex.utilities.utilities import get_class_properties, dict_str_load

from reV.handlers.exclusions import ExclusionLayers

from reVX.handlers.geotiff import Geotiff
from reVX.config.least_cost_xmission import LeastCostPathsConfig
from reVX.least_cost_xmission.config import XmissionConfig
from reVX.least_cost_xmission.least_cost_paths import (LeastCostPaths,
                                                       ReinforcementPaths)
from reVX.least_cost_xmission.least_cost_xmission import region_mapper
from reVX.least_cost_xmission.config.constants import (CELL_SIZE,
                                                       TRANS_LINE_CAT,
                                                       SUBSTATION_CAT,
                                                       BARRIERS_MULT,
                                                       BARRIER_H5_LAYER_NAME)
from reVX import __version__

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--name', '-n', default='LeastCostPaths', type=STR,
              show_default=True,
              help='Job name.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """
    Least Cost Paths Command Line Interface
    """
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose
    ctx.obj['NAME'] = name


@main.command()
def valid_config_keys():
    """
    Echo the valid Least Cost Paths config keys
    """
    click.echo(', '.join(get_class_properties(LeastCostPathsConfig)))


def run_local(ctx, config):
    """
    Run Least Cost Paths locally using config

    Parameters
    ----------
    ctx : click.ctx
        click ctx object
    config : reVX.config.least_cost_xmission.LeastCostPathsConfig
        Least Cost Paths config object.
    """
    ctx.obj['NAME'] = config.name
    ctx.invoke(local,
               cost_fpath=config.cost_fpath,
               features_fpath=config.features_fpath,
               cost_layers=config.cost_layers,
               network_nodes_fpath=config.network_nodes_fpath,
               transmission_lines_fpath=config.transmission_lines_fpath,
               xmission_config=config.xmission_config,
               capacity_class=config.capacity_class,
               clip_buffer=config.clip_buffer,
               start_index=0, step_index=1,
               tb_layer_name=config.tb_layer_name,
               barrier_mult=config.barrier_mult,
               max_workers=config.execution_control.max_workers,
               region_identifier_column=config.region_identifier_column,
               save_paths=config.save_paths,
               out_dir=config.dirout,
               log_dir=config.log_directory,
               ss_id_col=config.ss_id_col,
               verbose=config.log_level,
               li_cost_layers=config.length_invariant_cost_layers,
               tracked_layers=config.tracked_layers,
               cell_size=config.cell_size)


@main.command()
@click.option('--config', '-c', required=True,
              type=click.Path(exists=True),
              help='Filepath to AssemblyAreas config json file.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config, verbose):
    """
    Run Least Cost Paths from a config.
    """

    config = LeastCostPathsConfig(config)

    if 'VERBOSE' in ctx.obj:
        if any((ctx.obj['VERBOSE'], verbose)):
            config._log_level = logging.DEBUG
    elif verbose:
        config._log_level = logging.DEBUG

    if config.execution_control.option == 'local':
        run_local(ctx, config)
        return

    if config.execution_control.option not in {'eagle', 'kestrel'}:
        click.echo('Option "{}" is not supported'
                   .format(config.execution_control.option))
        return

    # No need to add index to file name
    if config.execution_control.nodes == 1:
        eagle(config)
        return

    name = config.name
    num_nodes = config.execution_control.nodes
    logger.info('Splitting features over {} SLURM jobs'.format(num_nodes))
    n_zfill = len(str(num_nodes))
    for i in range(num_nodes):
        config.name = '{}_{}'.format(name, str(i).zfill(n_zfill))
        eagle(config, start_index=i)


@main.command()
@click.option('--cost_fpath', '-cost', type=click.Path(exists=True),
              required=True,
              help=("Path to h5 file with cost rasters and other required "
                    "layers"))
@click.option('--features_fpath', '-feats', required=True,
              type=click.Path(exists=True),
              help="Path to GeoPackage with transmission features")
@click.option('--cost-layers', '-cl', required=True, multiple=True,
              default=(),
              help='Layer in H5 to add to total cost raster used for routing. '
                   'Multiple layers may be specified. If running '
                   'reinforcement computations, layer name may have '
                   'curly brackets (``{}``), which will be filled in '
                   'based on the capacity class input (e.g. '
                   '"tie_line_costs_{}MW")')
@click.option('--network_nodes_fpath', '-nn', type=STR, show_default=True,
              default=None,
              help=("Path to Network Nodes GeoPackage. If given alongside "
                    "`transmission_lines_fpath`, reinforcement path cost "
                    "calculation is run."))
@click.option('--transmission_lines_fpath', '-tl', type=STR, show_default=True,
              default=None,
              help=("Path to Transmission lines GeoPackage. This file can "
                    "contain other features, but transmission lines must "
                    "be identified by {!r}. If given alongside "
                    "`network_nodes_fpath`, reinforcement path cost "
                    "calculation is run.".format(TRANS_LINE_CAT)))
@click.option('--xmission_config', '-xcfg', type=STR, show_default=True,
              default=None,
              help=("Path to transmission config .json. Only used for "
                    "reinforcement path computations."))
@click.option('--capacity_class', '-cap', type=str, required=False,
              help=("Capacity class of the 'base' greenfield costs layer. "
                    "Costs will be scaled by the capacity corresponding to "
                    "this class to report reinforcement costs as $/MW. "
                    "Only used for reinforcement path computations."))
@click.option('--clip_buffer', '-cb', type=int,
              show_default=True, default=0,
              help="Optional number of array elements to buffer clip area by.")
@click.option('--start_index', '-start', type=int,
              show_default=True, default=0,
              help=("Start index of features to run."))
@click.option('--step_index', '-step', type=int,
              show_default=True, default=1,
              help=("Step index of features to run."))
@click.option('--tb_layer_name', '-tbln', default=BARRIER_H5_LAYER_NAME,
              type=STR, show_default=True,
              help='Name of transmission barrier layer in `cost_fpath` file. '
                   'This layer defines the multipliers applied to the cost '
                   'layer to determine LCP routing (but does not actually '
                   'affect output costs')
@click.option('--barrier_mult', '-bmult', type=float,
              show_default=True, default=BARRIERS_MULT,
              help=("Transmission barrier multiplier, used when computing the "
                    "least cost tie-line path"))
@click.option('--max_workers', '-mw', type=INT,
              show_default=True, default=None,
              help=("Number of workers to use for processing, if 1 run in "
                    "serial, if None use all available cores"))
@click.option('--region_identifier_column', '-rid', type=STR, default=None,
              help=("Name of column in reinforcement regions GeoPackage"
                    "containing a unique identifier for each region."))
@click.option('--save_paths', '-paths', is_flag=True,
              help="Flag to save least cost path as a multi-line geometry")
@click.option('--out_dir', '-o', type=STR, default='./',
              show_default=True,
              help='Directory to save least cost Paths values to.')
@click.option('--log_dir', '-log', default=None, type=STR,
              show_default=True,
              help='Directory to dump log files.')
@click.option('--ss_id_col', '-ssid', default="poi_gid", type=STR,
              show_default=True,
              help='Name of column used to unqiuely identify substations. '
                   'Used for reinforcement calcaultions only. ')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.option('--li-cost-layers', '-licl', required=False, multiple=True,
              default=(),
              help='Length-invariant cost layer in H5 to add to total cost '
                   'raster used for routing. These costs do not scale with '
                   'distance traversed acroiss the cell. Multiple layers may '
                   'be specified.')
@click.option('--tracked_layers', '-trl', type=STR, default=None,
              show_default=True,
              help=('Dictionary mapping layer names to strings, where the '
                    'strings are numpy methods that should be applied to '
                    'the layer along the LCP.'))
@click.option('--cell_size', '-cs', type=INT,
              show_default=True, default=CELL_SIZE,
              help=("Side length of a single cell in meters. Cells are "
                    "assumed to be square. Default is"))
@click.pass_context
def local(ctx, cost_fpath, features_fpath, cost_layers, network_nodes_fpath,
          transmission_lines_fpath, xmission_config, capacity_class,
          clip_buffer, start_index, step_index, tb_layer_name, barrier_mult,
          max_workers, region_identifier_column, save_paths, out_dir, log_dir,
          ss_id_col, verbose, li_cost_layers, tracked_layers, cell_size):
    """
    Run Least Cost Paths on local hardware
    """
    start_time = time.time()
    name = ctx.obj['NAME']
    if 'VERBOSE' in ctx.obj:
        verbose = any((ctx.obj['VERBOSE'], verbose))

    log_modules = [__name__, 'reVX', 'reV', 'rex']
    init_mult(name, log_dir, modules=log_modules, verbose=verbose)

    create_dirs(out_dir)
    logger.info('Computing Least Cost Paths connections and writing them to {}'
                .format(out_dir))

    logger.debug("Tracked layers input: %r", tracked_layers)
    if isinstance(tracked_layers, str):
        tracked_layers = dict_str_load(tracked_layers)
    cost_layers = list(cost_layers)
    li_cost_layers = list(li_cost_layers)

    is_reinforcement_run = (network_nodes_fpath is not None
                            and transmission_lines_fpath is not None)

    kwargs = {"clip_buffer": int(clip_buffer),
              "tb_layer_name": tb_layer_name,
              "barrier_mult": barrier_mult,
              "save_paths": save_paths,
              "max_workers": max_workers,
              "length_invariant_cost_layers": li_cost_layers,
              "tracked_layers": tracked_layers,
              "cell_size": cell_size}

    if is_reinforcement_run:
        logger.info('Detected reinforcement run!')
        xmission_config = XmissionConfig(config=xmission_config)
        kwargs["xmission_config"] = xmission_config
        cc_str = xmission_config._parse_cap_class(capacity_class)
        cap = xmission_config['power_classes'][cc_str]
        cost_layers = [layer.format(cap) for layer in cost_layers]
        logger.debug('Xmission Config: {}'.format(xmission_config))
        features = gpd.read_file(network_nodes_fpath)
        features, *__ = LeastCostPaths._map_to_costs(cost_fpath, features)
        kwargs["indices"] = features.index[start_index::step_index]
        kwargs["ss_id_col"] = ss_id_col
        least_costs = ReinforcementPaths.run(cost_fpath, features_fpath,
                                             network_nodes_fpath,
                                             region_identifier_column,
                                             transmission_lines_fpath,
                                             capacity_class, cost_layers,
                                             **kwargs)
    else:
        features = gpd.read_file(features_fpath)
        features, *__ = LeastCostPaths._map_to_costs(cost_fpath, features)
        kwargs["indices"] = features.index[start_index::step_index]
        least_costs = LeastCostPaths.run(cost_fpath, features_fpath,
                                         cost_layers, **kwargs)

    fpath_out = os.path.join(out_dir, f'{name}_lcp')
    if save_paths:
        fpath_out += '.gpkg'
        logger.info('Writing output to %s', fpath_out)
        least_costs.to_file(fpath_out, driver="GPKG", index=False)
    else:
        fpath_out += '.csv'
        logger.info('Writing output to %s', fpath_out)
        least_costs.to_csv(fpath_out, index=False)

    logger.info('Writing output complete')
    elapsed_time = (time.time() - start_time) / 60
    logger.info("Processing took %.2f minutes", elapsed_time)


@main.command()
@click.option('--features_fpath', '-feats', required=True,
              type=click.Path(exists=True),
              help="Path to GeoPackage with substation and transmission "
                   "features")
@click.option('--regions_fpath', '-regs', required=True,
              type=click.Path(exists=True),
              help=("Path to reinforcement regions GeoPackage."))
@click.option('--region_identifier_column', '-rid', required=True,
              type=STR,
              help=("Name of column in reinforcement regions GeoPackage"
                    "containing a unique identifier for each region."))
@click.option('--out_file', '-of', default=None, type=STR,
              help='Name for output GeoPackage file.')
@click.pass_context
def map_ss_to_rr(ctx, features_fpath, regions_fpath, region_identifier_column,
                 out_file):
    """
    Map substation locations to reinforcement regions.

    Reinforcement regions are user-defined. Typical regions are
    Balancing Areas, States, or Counties, though custom regions are also
    allowed. Each region must be supplied with a unique identifier in
    the input file.

    This method also removes substations that do not meet the min 69 kV
    voltage requirement and adds {'min_volts', 'max_volts'} fields to
    the remaining substations.

    .. Important:: This method DOES NOT clip the substations to the
      reinforcement regions boundary. All substations will be mapped to
      their closest region. It is your responsibility to remove any
      substations outside of the analysis region before calling this
      method.

    Doing the pre-processing step avoids any issues with substations
    being left out or double counted if they were simply clipped to the
    reinforcement region shapes.
    """
    log_level = "DEBUG" if ctx.obj.get('VERBOSE') else "INFO"
    init_logger('reVX', log_level=log_level)

    features = gpd.read_file(features_fpath)
    substations = (features[features.category == SUBSTATION_CAT]
                   .reset_index(drop=True)
                   .dropna(axis="columns", how="all"))

    regions = gpd.read_file(regions_fpath).to_crs(features.crs)
    logger.info("Mapping {:,d} substation locations to {:,d} "
                "reinforcement regions"
                .format(substations.shape[0], regions.shape[0]))

    map_func = region_mapper(regions, region_identifier_column)
    centroids = substations.centroid
    substations[region_identifier_column] = centroids.apply(map_func)

    logger.info("Calculating min/max voltage for each substation...")
    bad_subs = np.zeros(len(substations), dtype=bool)
    for idx, row in substations.iterrows():
        lines = row['trans_gids']
        if isinstance(lines, str):
            lines = json.loads(lines)

        lines_mask = features['gid'].isin(lines)
        voltage = features.loc[lines_mask, 'voltage'].values

        if np.max(voltage) >= 69:
            substations.loc[idx, 'min_volts'] = np.min(voltage)
            substations.loc[idx, 'max_volts'] = np.max(voltage)
        else:
            bad_subs[idx] = True

    if any(bad_subs):
        msg = ("The following sub-stations do not have the minimum "
               "required voltage of 69 kV and will be dropped:\n{}"
               .format(substations.loc[bad_subs, 'gid']))
        logger.warning(msg)
        warn(msg)
        substations = substations.loc[~bad_subs].reset_index(drop=True)

    logger.info("Writing substation output to {!r}".format(out_file))
    substations.to_file(out_file, driver="GPKG", index=False)


@main.command()
@click.option('--connections_fpath', '-con', required=True,
              type=click.Path(exists=True),
              help="Path to GeoPackage with substation and transmission "
                   "features")
@click.option('--region_identifier_column', '-rid', required=True,
              type=STR,
              help=("Name of column in reinforcement regions GeoPackage"
                    "containing a unique identifier for each region."))
@click.option('--out_file', '-of', required=True, type=STR,
              help='Name for output GeoPackage file.')
@click.pass_context
def ss_from_conn(ctx, connections_fpath, region_identifier_column, out_file):
    """Extract substations from connections table output by LCP.

    Substations extracted by this method can be usef for reinforcement
    calculations.

    This method also does minor filtering/formatting of the input
    connections table.

    Reinforcement regions are user-defined. Typical regions are
    Balancing Areas, States, or Counties, though custom regions are also
    allowed. Each region must be supplied with a unique identifier in
    the input file.

    .. Important:: This method DOES NOT clip the substations to the
      reinforcement regions boundary. All substations will be mapped to
      their closest region. It is your responsibility to remove any
      substations outside of the analysis region before calling this
      method.

    Doing the pre-processing step avoids any issues with substations
    being left out or double counted if they were simply clipped to the
    reinforcement region shapes.
    """
    log_level = "DEBUG" if ctx.obj.get('VERBOSE') else "INFO"
    init_logger('reVX', log_level=log_level)

    logger.info("Reading in connection info...")
    if connections_fpath.endswith(".csv"):
        connections = pd.read_csv(connections_fpath)
    elif connections_fpath.endswith(".gpkg"):
        connections = gpd.read_file(connections_fpath)
    else:
        raise ValueError("Unknown file ending for features file (must be "
                         f"'.csv' or '.gpkg'): {connections_fpath}")

    logger.info("Filtering out NaN's in connection info...")
    connections = connections[~connections["poi_gid"].isna()]
    connections = connections[~connections["poi_lat"].isna()]
    connections = connections[~connections["poi_lon"].isna()]

    logger.info("Extracting substation locations...")
    cols = ['poi_gid', 'poi_lat', 'poi_lon', region_identifier_column]
    poi_groups = connections.groupby(cols)
    pois = [info for info, __ in poi_groups]
    pois = pd.DataFrame(pois, columns=cols)

    geo = [Point(row["poi_lon"], row["poi_lat"])
           for __, row in pois.iterrows()]
    substations = gpd.GeoDataFrame(pois, crs="epsg:4326", geometry=geo)

    logger.info("Writing substation output to {!r}".format(out_file))
    substations["poi_gid"] = substations["poi_gid"].astype("Int64")
    substations.to_file(out_file, driver="GPKG", index=False)


@main.command()
@click.option('--network_nodes_fpath', '-nodes', type=click.Path(exists=True),
              help=("Path to network nodes GeoPackage. The "
                    "`region_identifier_column` will be added to this file if "
                    "it is missing."))
@click.option('--regions_fpath', '-regs', required=True,
              type=click.Path(exists=True),
              help=("Path to reinforcement regions GeoPackage. The "
                    "`region_identifier_column` from this file will be "
                    "added to the `network_nodes_fpath` if needed."))
@click.option('--region_identifier_column', '-rid', required=True,
              type=STR,
              help=("Name of column in reinforcement regions GeoPackage"
                    "containing a unique identifier for each region."))
@click.option('--crs_tempalate_file', '-ctf', default=None, type=STR,
              help=("Path to a fiel containing the CRS that should be used "
                    "to perform the mapping. This input can be an exclusions "
                    "HDF5 file, a GeoTIFF, or any file tha can be read with "
                    "GeoPandas. If `None`, the `network_nodes_fpath` file "
                    "CRS is used to perfom the mapping."))
@click.option('--out_file', '-of', default=None, type=STR,
              help='Name for output GeoPackage file. If `None`, the '
                   '`network_nodes_fpath` file will be overwritten. ')
@click.pass_context
def add_rr_to_nn(ctx, network_nodes_fpath, regions_fpath,
                 region_identifier_column, crs_tempalate_file, out_file):
    """Add reinforcement region column to network node file.

    Reinforcement regions are user-defined. Typical regions are
    Balancing Areas, States, or Counties, though custom regions are also
    allowed. Each region must be supplied with a unique identifier in
    the input file.
    """
    log_level = "DEBUG" if ctx.obj.get('VERBOSE') else "INFO"
    init_logger('reVX', log_level=log_level)
    crs_tempalate_file = Path(crs_tempalate_file or network_nodes_fpath)

    logger.info("Reading in network node info...")
    network_nodes_fpath = Path(network_nodes_fpath)
    network_nodes = gpd.read_file(network_nodes_fpath)

    logger.info("Reading in CRS template...")
    if crs_tempalate_file.suffix == ".h5":
        with ExclusionLayers(str(crs_tempalate_file)) as excl:
            crs = CRS.from_string(excl.crs)
    elif crs_tempalate_file.suffix in {".tif", ".tiff"}:
        with Geotiff(str(crs_tempalate_file)) as geo:
            crs = geo.profile["crs"]
    else:
        crs = gpd.read_file(crs_tempalate_file).crs

    network_nodes = network_nodes.to_crs(crs)
    regions = gpd.read_file(regions_fpath).to_crs(crs)
    if region_identifier_column in network_nodes:
        msg = ("Network nodes file {!r} was specified but it "
               "already contains the {!r} column. No data modified!"
               .format(str(network_nodes_fpath), region_identifier_column))
        logger.warning(msg)
        warn(msg)
        return

    logger.info("Adding regions to network nodes...")
    centroids = network_nodes.centroid
    map_func = region_mapper(regions, region_identifier_column)
    network_nodes[region_identifier_column] = centroids.apply(map_func)

    if out_file is None:
        out_fn = "{}.gpkg".format(network_nodes_fpath.stem)
        out_file = network_nodes_fpath.parent / out_fn

    logger.info("Writing updated network node data to {!r}"
                .format(str(out_file)))
    network_nodes.to_file(out_file, driver="GPKG")


def get_node_cmd(config, start_index=0):
    """
    Get the node CLI call for Least Cost Paths

    Parameters
    ----------
    config : reVX.config.least_cost_xmission.LeastCostPathsConfig
        Least Cost Paths config object.

    Returns
    -------
    cmd : str
        CLI call to submit to SLURM execution.
    """

    args = ['-n {}'.format(SLURM.s(config.name)),
            'local',
            '-cost {}'.format(SLURM.s(config.cost_fpath)),
            '-feats {}'.format(SLURM.s(config.features_fpath)),
            '-nn {}'.format(SLURM.s(config.network_nodes_fpath)),
            '-tl {}'.format(SLURM.s(config.transmission_lines_fpath)),
            '-rid {}'.format(SLURM.s(config.region_identifier_column)),
            '-cap {}'.format(SLURM.s(config.capacity_class)),
            '-cb {}'.format(SLURM.s(config.clip_buffer)),
            '-start {}'.format(SLURM.s(start_index)),
            '-step {}'.format(SLURM.s(config.execution_control.nodes or 1)),
            '-tbln {}'.format(SLURM.s(config.tb_layer_name)),
            '-bmult {}'.format(SLURM.s(config.barrier_mult)),
            '-ssid {}'.format(SLURM.s(config.ss_id_col)),
            '-mw {}'.format(SLURM.s(config.execution_control.max_workers)),
            '-o {}'.format(SLURM.s(config.dirout)),
            '-log {}'.format(SLURM.s(config.log_directory)),
            '-cs {}'.format(SLURM.s(config.cell_size)),
            ]

    for layer in config.cost_layers:
        args.append(f'-cl {layer}')
    for layer in config.length_invariant_cost_layers:
        args.append(f'-licl {layer}')

    if config.save_paths:
        args.append('-paths')

    if config.tracked_layers:
        args.append('-trl {}'.format(SLURM.s(config.tracked_layers)))

    if config.log_level == logging.DEBUG:
        args.append('-v')

    if config.xmission_config is not None:
        args.append('-xcfg {}'.format(SLURM.s(config.xmission_config)),)

    cmd = ('python -m reVX.least_cost_xmission.least_cost_paths_cli {}'
           .format(' '.join(args)))
    logger.debug('Submitting the following cli call:\n\t{}'.format(cmd))

    return cmd


def eagle(config, start_index=0):
    """
    Run Least Cost Paths on Eagle HPC.

    Parameters
    ----------
    config : reVX.config.least_cost_xmission.LeastCostPathsConfig
        Least Cost Paths config object.
    """

    cmd = get_node_cmd(config, start_index)
    name = config.name
    log_dir = config.log_directory
    stdout_path = os.path.join(log_dir, 'stdout/')

    slurm_manager = SLURM()

    logger.info('Running Least Cost Paths on Eagle with '
                'node name "{}"'.format(name))
    out = slurm_manager.sbatch(cmd,
                               alloc=config.execution_control.allocation,
                               memory=config.execution_control.memory,
                               walltime=config.execution_control.walltime,
                               feature=config.execution_control.feature,
                               name=name, stdout_path=stdout_path,
                               conda_env=config.execution_control.conda_env,
                               module=config.execution_control.module)[0]
    if out:
        msg = ('Kicked off Least Cost Paths "{}" '
               '(SLURM jobid #{}) on Eagle.'
               .format(name, out))
    else:
        msg = ('Was unable to kick off Least Cost Paths '
               '"{}". Please see the stdout error messages'
               .format(name))

    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running Least Cost Paths CLI')
        raise
