# -*- coding: utf-8 -*-
# pylint: disable=all
"""
Least Cost Xmission Command Line Interface

TODO - add cmd line doc

"""
import os
import sys
import time
import click
import logging
import warnings
import pandas as pd
import geopandas as gpd
from typing import List

from rex.utilities.loggers import init_mult, create_dirs, init_logger
from rex.utilities.cli_dtypes import STR, INTLIST, INT, FLOAT
from rex.utilities.hpc import SLURM
from rex.utilities.utilities import get_class_properties, dict_str_load

from reVX import __version__
from reVX.config.least_cost_xmission import LeastCostXmissionConfig
from reVX.least_cost_xmission.least_cost_xmission import (LeastCostXmission,
                                                          RegionalXmission)
from reVX.least_cost_xmission.config.constants import (CELL_SIZE,
                                                       TRANS_LINE_CAT,
                                                       LOAD_CENTER_CAT,
                                                       SINK_CAT,
                                                       SUBSTATION_CAT,
                                                       MINIMUM_SPUR_DIST_KM,
                                                       BARRIERS_MULT,
                                                       CLIP_RASTER_BUFFER,
                                                       NUM_NN_SINKS,
                                                       BARRIER_H5_LAYER_NAME,
                                                       ISO_H5_LAYER_NAME)
from reVX.least_cost_xmission.least_cost_paths import min_reinforcement_costs

TRANS_CAT_TYPES = [TRANS_LINE_CAT, LOAD_CENTER_CAT, SINK_CAT, SUBSTATION_CAT]

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--name', '-n', default='LeastCostXmission', type=STR,
              show_default=True,
              help='Job name.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """
    Least Cost Xmission Command Line Interface
    """
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose
    ctx.obj['NAME'] = name


@main.command()
def valid_config_keys():
    """
    Echo the valid Least Cost Xmission config keys
    """
    click.echo(', '.join(get_class_properties(LeastCostXmissionConfig)))


@main.command()
@click.option('--config', '-c', required=True,
              type=click.Path(exists=True),
              help='Filepath to Least Cost Xmission config json file.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config, verbose):
    """
    Run Least Cost Xmission from a config.
    """

    config = LeastCostXmissionConfig(config)
    option = config.execution_control.option

    if 'VERBOSE' in ctx.obj:
        if any((ctx.obj['VERBOSE'], verbose)):
            config._log_level = logging.DEBUG
    elif verbose:
        config._log_level = logging.DEBUG

    if option == 'local':
        run_local(ctx, config)
        return

    if option not in {'eagle', 'kestrel'}:
        click.echo('Option "{}" is not supported'.format(option))
        return

    if config.execution_control.nodes == 1:
        eagle(config, config.sc_point_gids)
        return

    # Split gids over mulitple SLURM jobs
    name = config.name
    logger.info('Splitting SC points over {} SLURM jobs'
                .format(config.execution_control.nodes))
    for i in range(config.execution_control.nodes):
        config.name = '{}_{}'.format(name, i)
        eagle(config, config.sc_point_gids[i::config.execution_control.nodes])


@main.command()
@click.option('--cost_fpath', '-cost', type=click.Path(exists=True),
              required=True,
              help=("Path to h5 file with cost rasters and other required "
                    "layers"))
@click.option('--features_fpath', '-feats', required=True,
              type=click.Path(exists=True),
              help="Path to GeoPackage with transmission features")
@click.option('--regions_fpath', '-regs', type=STR, show_default=True,
              default=None,
              help=("Path to reinforcement regions GeoPackage. If not `None`, "
                    "Least Cost Xmission is run with reinforcement path "
                    "costs. Features must be substations only, and the "
                    "substation file must contain a "
                    "`region_identifier_column` column that matches the "
                    "`region_identifier_column` ID in this file for the "
                    "reinforcement region containing that substation. "))
@click.option('--region_identifier_column', '-rid', type=STR, default=None,
              help=("Name of column in reinforcement regions GeoPackage"
                    "containing a unique identifier for each region."))
@click.option('--capacity_class', '-cap', type=str, required=True,
              help=("Capacity class of transmission features to connect "
                    "supply curve points to"))
@click.option('--resolution', '-res', type=int,
              show_default=True, default=128,
              help=("SC point resolution"))
@click.option('--xmission_config', '-xcfg', type=STR, show_default=True,
              default=None,
              help=("Path to Xmission config .json"))
@click.option('--min_line_length', '-mll', type=int,
              show_default=True, default=MINIMUM_SPUR_DIST_KM,
              help=("Minimum Tie-line length."))
@click.option('--sc_point_gids', '-gids', type=INTLIST, show_default=True,
              default=None, help=("List of sc_point_gids to connect to. If "
                                  "running `from_config`, this can also be a "
                                  "path to a CSV file with a 'sc_point_gid' "
                                  "column containing the GID's to run. Note "
                                  "the missing 's' in the column name - this "
                                  "makes it seamless to run on a supply curve "
                                  "output from reV"))
@click.option('--nn_sinks', '-nn', type=int,
              show_default=True, default=NUM_NN_SINKS,
              help=("Number of nearest neighbor sinks to use for clipping "
                    "radius calculation. This is overridden by --radius"))
@click.option('--clipping_buffer', '-buffer', type=float,
              show_default=True, default=CLIP_RASTER_BUFFER,
              help=("Buffer to expand clipping radius by"))
@click.option('--tb_layer_name', '-tbln', default=BARRIER_H5_LAYER_NAME,
              type=STR, show_default=True,
              help='Name of transmission barrier layer in `cost_fpath` file. '
                   'This layer defines the multipliers applied to the cost '
                   'layer to determine LCP routing (but does not actually '
                   'affect output costs')
@click.option('--barrier_mult', '-bmult', type=float,
              show_default=True, default=BARRIERS_MULT,
              help=("Tranmission barrier multiplier, used when computing the "
                    "least cost tie-line path"))
@click.option('--iso_regions_layer_name', '-irln', default=ISO_H5_LAYER_NAME,
              type=STR, show_default=True,
              help='Name of ISO regions layer in `cost_fpath` file. The '
                   "layer maps pixels to ISO region ID's (1, 2, 3, 4, etc.) .")
@click.option('--max_workers', '-mw', type=INT,
              show_default=True, default=None,
              help=("Number of workers to use for processing, if 1 run in "
                    "serial, if None use all available cores"))
@click.option('--out_dir', '-o', type=STR, default='./out',
              show_default=True,
              help='Directory to save least cost xmission values to.')
@click.option('--log_dir', '-log', default=None, type=STR,
              show_default=True,
              help='Directory to dump log files.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.option('--save_paths', is_flag=True,
              help='Save least cost paths and data to GeoPackage.')
@click.option('--radius', '-rad', type=INT,
              show_default=True, default=None,
              help=("Radius to clip costs raster to in pixels This overrides "
                    "--nn_sinks if set."))
@click.option('--mp_delay', '-mpd', type=FLOAT, show_default=True, default=3.0,
              help=("Delay in seconds between starting multi-process workers. "
                    "Useful for reducing memory spike at working startup."))
@click.option('--expand_radius', '-er', is_flag=True,
              help='Flag to expand radius until at least one transmission '
                   'feature is included for connection. Has no effect if '
                   'radius input is ``None``.')
@click.option('--simplify-geo', type=FLOAT,
              show_default=True, default=None,
              help=("Simplify path geometries by a value before writing to "
                    "GeoPackage."))
@click.option('--cost-layers', '-cl', required=True, multiple=True,
              default=(),
              help='Layer in H5 to add to total cost raster used for routing. '
                   'Multiple layers may be specified. Layer name may have '
                   'curly brackets (``{}``), which will be filled in '
                   'based on the capacity class input (e.g. '
                   '"tie_line_costs_{}MW")')
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
@click.option('--length_mult_kind', '-lmk', type=STR, default='linear',
              show_default=True,
              help='Type of length multiplier calcualtion. "step" computes '
                   'length multipliers using a step function, while "linear" '
                   'computes the length multiplier using a linear '
                   'interpolation between 0 amd 10 mile spur-line lengths.')
@click.option('--cell_size', '-cs', type=INT,
              show_default=True, default=CELL_SIZE,
              help=("Side length of a single cell in meters. Cells are "
                    "assumed to be square. Default is"))
@click.pass_context
def local(ctx, cost_fpath, features_fpath, regions_fpath,
          region_identifier_column, capacity_class, resolution,
          xmission_config, min_line_length, sc_point_gids, nn_sinks,
          clipping_buffer, tb_layer_name, barrier_mult,
          iso_regions_layer_name, max_workers, out_dir, log_dir, verbose,
          save_paths, radius, expand_radius, mp_delay, simplify_geo,
          cost_layers: List[str], li_cost_layers, tracked_layers,
          length_mult_kind, cell_size):
    """
    Run Least Cost Xmission on local hardware
    """
    start_time = time.time()
    name = ctx.obj['NAME']
    if 'VERBOSE' in ctx.obj:
        verbose = any((ctx.obj['VERBOSE'], verbose))

    log_modules = [__name__, 'reVX', 'reV', 'rex']
    init_mult(name, log_dir, modules=log_modules, verbose=verbose)

    create_dirs(out_dir)
    logger.info('Computing Least Cost Xmission connections and writing them {}'
                .format(out_dir))

    cost_layers = list(cost_layers)
    li_cost_layers = list(li_cost_layers)
    if isinstance(tracked_layers, str):
        tracked_layers = dict_str_load(tracked_layers)

    kwargs = {"resolution": resolution,
              "xmission_config": xmission_config,
              "min_line_length": min_line_length,
              "sc_point_gids": sc_point_gids,
              "clipping_buffer": clipping_buffer,
              "tb_layer_name": tb_layer_name,
              "barrier_mult": barrier_mult,
              "iso_regions_layer_name": iso_regions_layer_name,
              "max_workers": max_workers,
              "save_paths": save_paths,
              "simplify_geo": simplify_geo,
              "radius": radius,
              "expand_radius": expand_radius,
              "mp_delay": mp_delay,
              "length_invariant_cost_layers": li_cost_layers,
              "length_mult_kind": length_mult_kind,
              "tracked_layers": tracked_layers,
              "cell_size": cell_size}

    if regions_fpath is not None:
        least_costs = RegionalXmission.run(cost_fpath, features_fpath,
                                           regions_fpath,
                                           region_identifier_column,
                                           capacity_class, cost_layers,
                                           **kwargs)
    else:
        kwargs["nn_sinks"] = nn_sinks
        least_costs = LeastCostXmission.run(cost_fpath, features_fpath,
                                            capacity_class, cost_layers,
                                            **kwargs)
    if len(least_costs) == 0:
        logger.error('No paths found.')
        return

    ext = 'gpkg' if save_paths else 'csv'
    fn_out = '{}_{}_{}.{}'.format(name, capacity_class, resolution, ext)
    fpath_out = os.path.join(out_dir, fn_out)

    logger.info('Writing output to %s', fpath_out)
    if save_paths:
        least_costs.to_file(fpath_out, driver='GPKG')
    else:
        least_costs.to_csv(fpath_out, index=False)

    logger.info('Writing output complete')
    elapsed_time = (time.time() - start_time) / 60
    logger.info("Processing took %.2f minutes", elapsed_time)


@main.command()
@click.option('--split-to-geojson', '-s', is_flag=True,
              help='After merging GeoPackages, split into GeoJSON by POI name'
              '.')
@click.option('--suppress-combined-file', is_flag=True,
              help='Don\'t create combined layer.')
@click.option('--out-file', '-of', default=None, type=STR,
              help='Name for output GeoPackage/CSV file.')
@click.option('--drop', '-d', default=None, type=STR, multiple=True,
              help=('Transmission feature category types to drop from '
                    'results. Options: {}'.format(", ".join(TRANS_CAT_TYPES))))
@click.option('--out-dir', '-od', type=click.Path(),
              default='./out', show_default=True,
              help='Output directory for output files.')
@click.option('--simplify-geo', type=FLOAT,
              show_default=True, default=None,
              help='Simplify path geometries by a value before exporting.')
@click.option('--ss_id_col', '-ssid', default=None, type=STR,
              show_default=True,
              help='Name of column used to unqiuely identify substations. '
                   'Used for reinforcement calcaultions only. ')
@click.argument('files', type=STR, nargs=-1)
@click.pass_context
# flake8: noqa: C901
def merge_output(ctx, split_to_geojson, suppress_combined_file, out_file,
                 out_dir, drop, simplify_geo, ss_id_col, files):
    """
    Merge output GeoPackage/CSV files and optionally convert to GeoJSON
    """
    log_level = "DEBUG" if ctx.obj.get('VERBOSE') else "INFO"
    init_logger('reVX', log_level=log_level)

    if len(files) == 0:
        logger.error('No files passed to be merged')
        return

    if drop:
        for cat in drop:
            if cat not in TRANS_CAT_TYPES:
                logger.info('--drop options must be one or more of %s, '
                            'received %s', TRANS_CAT_TYPES, drop)
                return

    warnings.filterwarnings('ignore', category=RuntimeWarning)
    dfs = []
    for i, file in enumerate(files, start=1):
        logger.info('Loading %s (%i/%i)', file, i, len(files))
        df_tmp = gpd.read_file(file) if "gpkg" in file else pd.read_csv(file)
        dfs.append(df_tmp)

    df = pd.concat(dfs)
    warnings.filterwarnings('default', category=RuntimeWarning)

    if drop:
        mask = df['category'].isin(drop)
        logger.info('Dropping %d of %d total features with category(ies): %s',
                    mask.sum(), len(df), ", ".join(drop))
        df = df[~mask]

    df = df.reset_index()

    if len(df) == 0:
        logger.info('No transmission features to save.')
        return

    if simplify_geo:
        logger.info('Simplifying geometries by {}'.format(simplify_geo))
        df.geometry = df.geometry.simplify(simplify_geo)

    if ss_id_col is not None:
        logger.info('Computing minimum reinforcement cost')
        df = min_reinforcement_costs(df, group_col=ss_id_col)

    create_dirs(out_dir)

    # Create combined output file
    if not suppress_combined_file:
        out_file = ('combo_{}'.format(files[0])
                    if out_file is None else out_file)
        out_file = os.path.join(out_dir, out_file)
        logger.info('Saving all combined paths to %s', out_file)
        if "gpkg" in out_file:
            df.to_file(out_file, driver="GPKG")
        else:
            df.to_csv(out_file, index=False)

    # Split out put in to GeoJSON by POI name
    if split_to_geojson:
        if not isinstance(df, gpd.GeoDataFrame):
            click.echo('Geo-spatial aware input files must be provided to '
                       'split to Geo-JSON.')
            sys.exit(1)
        pois = set(df['POI Name'])
        for i, poi in enumerate(pois, start=1):
            safe_poi_name = poi.replace(' ', '_').replace('/', '_')
            out_file = os.path.join(out_dir, f"{safe_poi_name}_paths.geojson")
            paths = df[df['POI Name'] == poi].to_crs(epsg=4326)
            logger.info('Writing {} paths for {} to {} ({}/{})'
                        .format(len(paths), poi, out_file, i, len(pois)))
            paths.to_file(out_file, driver="GeoJSON")


@main.command()
@click.option('--cost_fpath', '-f', required=True,
              type=click.Path(exists=True),
              help=("Path to GeoPackage/CSV file with calculated transmission "
                    "costs. This file must have a 'trans_gid' column that "
                    "will be used to merge in the reinforcement costs."))
@click.option('--reinforcement_cost_fpath', '-r', required=True,
              type=click.Path(exists=True),
              help=("Path to GeoPackage/CSV file with calculated "
                    "reinforcement costs. This file must have a 'gid' column "
                    "that will be used to merge in the reinforcement costs."))
@click.option('--merge_column', '-mc', default="trans_gid", type=STR,
              help=("Name of column in `cost_fpath` and "
                    "`reinforcement_cost_fpath` files to merge on."))
@click.option('--out_file', '-of', default=None, type=STR,
              help='Name for output GeoPackage/CSV file.')
@click.pass_context
def merge_reinforcement_costs(ctx, cost_fpath, reinforcement_cost_fpath,
                              merge_column, out_file):
    """
    Merge reinforcement costs into transmission costs.
    """
    log_level = "DEBUG" if ctx.obj.get('VERBOSE') else "INFO"
    init_logger('reVX', log_level=log_level)
    logger.info("Merging reinforcement costs into transmission costs...")

    logger.debug("Reading in transmission costs from %s", str(cost_fpath))
    costs = (gpd.read_file(cost_fpath)
             if "gpkg" in cost_fpath
             else pd.read_csv(cost_fpath))

    logger.debug("Reading in reinforcement costs from %s",
                 str(reinforcement_cost_fpath))
    r_costs = (gpd.read_file(reinforcement_cost_fpath)
               if "gpkg" in reinforcement_cost_fpath
               else pd.read_csv(reinforcement_cost_fpath))

    logger.info("Loaded spur-line costs for %d substations and "
                "reinforcement costs for %d substations",
                len(costs[merge_column].unique()),
                len(r_costs[merge_column].unique()))

    r_costs.index = r_costs[merge_column].values
    costs = costs[costs[merge_column].isin(r_costs[merge_column])].copy()

    logger.info("Found %d substations with both spur-line and "
                "reinforcement costs",
                len(costs[merge_column].unique()))

    r_cols = ["reinforcement_poi_lat", "reinforcement_poi_lon",
              "reinforcement_dist_km", "reinforcement_cost_per_mw"]
    costs[r_cols] = r_costs.loc[costs[merge_column], r_cols].values

    logger.info("Writing output to {!r}".format(out_file))

    if "gpkg" in out_file:
        costs.to_file(out_file, driver="GPKG", index=False)
    else:
        costs.to_csv(out_file, index=False)


def get_node_cmd(config, gids):
    """
    Get the node CLI call for Least Cost Xmission

    Parameters
    ----------
    config : reVX.config.least_cost_xmission.LeastCostXmissionConfig
        Least Cost Xmission config object.
    gids : list
        List of SC point GID values to submit to local command.

    Returns
    -------
    cmd : str
        CLI call to submit to SLURM execution.
    """
    args = ['-n {}'.format(SLURM.s(config.name)),
            'local',
            '-cost {}'.format(SLURM.s(config.cost_fpath)),
            '-feats {}'.format(SLURM.s(config.features_fpath)),
            '-regs {}'.format(SLURM.s(config.regions_fpath)),
            '-rid {}'.format(SLURM.s(config.region_identifier_column)),
            '-cap {}'.format(SLURM.s(config.capacity_class)),
            '-res {}'.format(SLURM.s(config.resolution)),
            '-xcfg {}'.format(SLURM.s(config.xmission_config)),
            '-mll {}'.format(SLURM.s(config.min_line_length)),
            '-gids {}'.format(SLURM.s(gids)),
            '-nn {}'.format(SLURM.s(config.nn_sinks)),
            '-buffer {}'.format(SLURM.s(config.clipping_buffer)),
            '-tbln {}'.format(SLURM.s(config.tb_layer_name)),
            '-bmult {}'.format(SLURM.s(config.barrier_mult)),
            '-irln {}'.format(SLURM.s(config.iso_regions_layer_name)),
            '-mw {}'.format(SLURM.s(config.execution_control.max_workers)),
            '-o {}'.format(SLURM.s(config.dirout)),
            '-log {}'.format(SLURM.s(config.log_directory)),
            '-lmk {}'.format(SLURM.s(config.length_mult_kind)),
            '-mpd {}'.format(SLURM.s(config.mp_delay)),
            '-cs {}'.format(SLURM.s(config.cell_size)),
            ]

    for layer in config.cost_layers:
        args.append(f'-cl {layer}')
    for layer in config.length_invariant_cost_layers:
        args.append(f'-licl {layer}')

    if config.save_paths:
        args.append('--save_paths')
    if config.radius:
        args.append('-rad {}'.format(config.radius))
    if config.expand_radius:
        args.append('-er')
    if config.simplify_geo:
        args.append('--simplify-geo {}'.format(config.simplify_geo))
    if config.tracked_layers:
        args.append('-trl {}'.format(SLURM.s(config.tracked_layers)))

    if config.log_level == logging.DEBUG:
        args.append('-v')

    cmd = ('python -m reVX.least_cost_xmission.least_cost_xmission_cli {}'
           .format(' '.join(args)))
    logger.debug('Submitting the following cli call:\n\t{}'.format(cmd))

    return cmd


def run_local(ctx, config):
    """
    Run Least Cost Xmission locally using config

    Parameters
    ----------
    ctx : click.ctx
        click ctx object
    config : reVX.config.least_cost_xmission.LeastCostXmissionConfig
        Least Cost Xmission config object.
    """
    ctx.obj['NAME'] = config.name
    ctx.invoke(local,
               cost_fpath=config.cost_fpath,
               features_fpath=config.features_fpath,
               regions_fpath=config.regions_fpath,
               capacity_class=config.capacity_class,
               resolution=config.resolution,
               xmission_config=config.xmission_config,
               min_line_length=config.min_line_length,
               sc_point_gids=config.sc_point_gids,
               nn_sinks=config.nn_sinks,
               clipping_buffer=config.clipping_buffer,
               tb_layer_name=config.tb_layer_name,
               barrier_mult=config.barrier_mult,
               iso_regions_layer_name=config.iso_regions_layer_name,
               region_identifier_column=config.region_identifier_column,
               max_workers=config.execution_control.max_workers,
               out_dir=config.dirout,
               log_dir=config.log_directory,
               verbose=config.log_level,
               radius=config.radius,
               expand_radius=config.expand_radius,
               mp_delay=config.mp_delay,
               save_paths=config.save_paths,
               simplify_geo=config.simplify_geo,
               cost_layers=config.cost_layers,
               li_cost_layers=config.length_invariant_cost_layers,
               tracked_layers=config.tracked_layers,
               length_mult_kind=config.length_mult_kind,
               cell_size=config.cell_size)


def eagle(config, gids):
    """
    Run Least Cost Xmission on Eagle HPC.

    Parameters
    ----------
    config : reVX.config.least_cost_xmission.LeastCostXmissionConfig
        Least Cost Xmission config object.
    """
    init_logger('rex', log_level='DEBUG')

    cmd = get_node_cmd(config, gids)
    name = config.name
    log_dir = config.log_directory
    stdout_path = os.path.join(log_dir, 'stdout/')

    slurm_manager = SLURM()

    logger.info('Running Least Cost Xmission on Eagle with '
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
        msg = ('Kicked off Least Cost Xmission "{}" '
               '(SLURM jobid #{}) on Eagle.'
               .format(name, out))
    else:
        msg = ('Was unable to kick off Least Cost Xmission '
               '"{}". Please see the stdout error messages'
               .format(name))

    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running Least Cost Xmission CLI')
        raise
