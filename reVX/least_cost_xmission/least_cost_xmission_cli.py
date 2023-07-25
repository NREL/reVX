# -*- coding: utf-8 -*-
# pylint: disable=all
"""
Least Cost Xmission Command Line Interface

TODO - add cmd line doc

"""
import os
import click
import logging
import warnings
import pandas as pd
import geopandas as gpd
from pathlib import Path

from rex.utilities.loggers import init_mult, create_dirs, init_logger
from rex.utilities.cli_dtypes import STR, INTLIST, INT, FLOAT
from rex.utilities.hpc import SLURM
from rex.utilities.utilities import get_class_properties

from reV.supply_curve.extent import SupplyCurveExtent
from reVX import __version__
from reVX.config.least_cost_xmission import LeastCostXmissionConfig
from reVX.least_cost_xmission.least_cost_xmission import (LeastCostXmission,
                                                          ReinforcedXmission)
from reVX.least_cost_xmission.config import (TRANS_LINE_CAT, LOAD_CENTER_CAT,
                                             SINK_CAT, SUBSTATION_CAT)
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
              help='Filepath to AssemblyAreas config json file.')
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

    if option != 'eagle':
        click.echo('Option "{}" is not supported'.format(option))
        return

    if config.execution_control.nodes == 1:
        eagle(config)
        return

    # Split gids over mulitple SLURM jobs
    name = config.name
    logger.info('Splitting SC points over {} SLURM jobs'
                .format(config.execution_control.nodes))
    for i in range(config.execution_control.nodes):
        config.name = '{}_{}'.format(name, i)
        eagle(config, start_index=i)


@main.command()
@click.option('--cost_fpath', '-cost', type=click.Path(exists=True),
              required=True,
              help=("Path to h5 file with cost rasters and other required "
                    "layers"))
@click.option('--features_fpath', '-feats', required=True,
              type=click.Path(exists=True),
              help="Path to GeoPackage with transmission features")
@click.option('--balancing_areas_fpath', '-ba', type=STR, show_default=True,
              default=None,
              help=("Path to Balancing areas GeoPackage. If no `None`, "
                    "Least Cost Xmission is run with reinforcement path "
                    "costs. Features must be substations only, and the "
                    "substation file must contain a 'ba_str' column that "
                    "matches the BA ID in this file for the balancing area "
                    "containing that substation. "))
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
              show_default=True, default=0,
              help=("Minimum Tie-line length."))
@click.option('--sc_point_start_index', '-start', type=int,
              show_default=True, default=0,
              help=("Start index of supply curve points to run."))
@click.option('--sc_point_step_index', '-step', type=int,
              show_default=True, default=1,
              help=("Step index of supply curve points to run."))
@click.option('--nn_sinks', '-nn', type=int,
              show_default=True, default=2,
              help=("Number of nearest neighbor sinks to use for clipping "
                    "radius calculation. This is overridden by --radius"))
@click.option('--clipping_buffer', '-buffer', type=float,
              show_default=True, default=1.05,
              help=("Buffer to expand clipping radius by"))
@click.option('--barrier_mult', '-bmult', type=float,
              show_default=True, default=100,
              help=("Tranmission barrier multiplier, used when computing the "
                    "least cost tie-line path"))
@click.option('--state_connections', '-acws', is_flag=True,
              help='Flag to allow substations ot connect to any endpoints '
                   'within their state. Default is not verbose.')
@click.option('--max_workers', '-mw', type=INT,
              show_default=True, default=None,
              help=("Number of workers to use for processing, if 1 run in "
                    "serial, if None use all available cores"))
@click.option('--out_dir', '-o', type=STR, default='./',
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
@click.option('--simplify-geo', type=FLOAT,
              show_default=True, default=None,
              help=("Simplify path geometries by a value before writing to "
                    "GeoPackage."))
@click.pass_context
def local(ctx, cost_fpath, features_fpath, balancing_areas_fpath,
          capacity_class, resolution, xmission_config, min_line_length,
          sc_point_start_index, sc_point_step_index, nn_sinks,
          clipping_buffer, barrier_mult, state_connections, max_workers,
          out_dir, log_dir, verbose, save_paths, radius, simplify_geo):
    """
    Run Least Cost Xmission on local hardware
    """
    name = ctx.obj['NAME']
    if 'VERBOSE' in ctx.obj:
        verbose = any((ctx.obj['VERBOSE'], verbose))

    log_modules = [__name__, 'reVX', 'reV', 'rex']
    init_mult(name, log_dir, modules=log_modules, verbose=verbose)

    create_dirs(out_dir)
    logger.info('Computing Least Cost Xmission connections and writing them {}'
                .format(out_dir))
    sce = SupplyCurveExtent(cost_fpath, resolution=resolution)
    sc_point_gids = list(sce.points.index.values)
    sc_point_gids = sc_point_gids[sc_point_start_index::sc_point_step_index]
    kwargs = {"resolution": resolution,
              "xmission_config": xmission_config,
              "min_line_length": min_line_length,
              "sc_point_gids": sc_point_gids,
              "clipping_buffer": clipping_buffer,
              "barrier_mult": barrier_mult,
              "max_workers": max_workers,
              "save_paths": save_paths,
              "simplify_geo": simplify_geo,
              "radius": radius}
    if balancing_areas_fpath is not None:
        kwargs["allow_connections_within_states"] = state_connections
        least_costs = ReinforcedXmission.run(cost_fpath, features_fpath,
                                             balancing_areas_fpath,
                                             capacity_class, **kwargs)
    else:
        kwargs["nn_sinks"] = nn_sinks
        least_costs = LeastCostXmission.run(cost_fpath, features_fpath,
                                            capacity_class, **kwargs)
    if len(least_costs) == 0:
        logger.error('No paths found.')
        return

    ext = 'gpkg' if save_paths else 'csv'
    fn_out = '{}_{}_{}.{}'.format(name, capacity_class, resolution, ext)
    fpath_out = os.path.join(out_dir, fn_out)

    logger.info('Writing output to {}'.format(fpath_out))
    if save_paths:
        least_costs.to_file(fpath_out, driver='GPKG')
    else:
        least_costs.to_csv(fpath_out, index=False)
    logger.info('Writing output complete')


@main.command()
@click.option('--split-to-geojson', '-s', is_flag=True,
              help='After merging GeoPackages, split into GeoJSON by POI name'
              '.')
@click.option('--out-file', '-of', default=None, type=STR,
              help='Name for output GeoPackage file.')
@click.option('--drop', '-d', default=None, type=STR, multiple=True,
              help=('Transmission feature category types to drop from '
                    'results. Options: {}'.format(", ".join(TRANS_CAT_TYPES))))
@click.option('--out-dir', '-od', type=click.Path(exists=True),
              default='.', show_default=True,
              help='Output directory for output files. Path must exist.')
@click.option('--simplify-geo', type=FLOAT,
              show_default=True, default=None,
              help='Simplify path geometries by a value before exporting.')
@click.argument('files', type=STR, nargs=-1)
@click.pass_context
def merge_output(ctx, split_to_geojson, out_file, out_dir, drop,  # noqa
                 simplify_geo, files):
    """
    Merge output GeoPackage/CSV files and optionally convert to GeoJSON
    """
    log_level = "DEBUG" if ctx.obj.get('VERBOSE') else "INFO"
    init_logger('reVX', log_level=log_level)

    if len(files) == 0:
        logger.info('No files passed to be merged')
        return

    if len(files) == 1:
        files = sorted(Path(out_dir).glob(files[0]))

    logger.debug('Merging {}'.format(files))

    if drop:
        for cat in drop:
            if cat not in TRANS_CAT_TYPES:
                logger.info('--drop options must on or more of {}, received {}'
                            .format(TRANS_CAT_TYPES, drop))
                return

    logger.info('Loading: {}'.format(", ".join(files)))
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    df = pd.concat([gpd.read_file(f) if "gpkg" in f else pd.read_csv(f)
                    for f in files])
    warnings.filterwarnings('default', category=RuntimeWarning)

    if drop:
        mask = df['category'].isin(drop)
        logger.info('Dropping {} of {} total features with category(ies): {}'
                    .format(mask.sum(), len(df), ", ".join(drop)))
        df = df[~mask]

    df = df.reset_index()

    if len(df) == 0:
        logger.info('No transmission features to save.')
        return

    if simplify_geo:
        logger.info('Simplifying geometries by {}'.format(simplify_geo))
        df.geometry = df.geometry.simplify(simplify_geo)

    if all(col in df for col in ["gid", "reinforcement_cost_per_mw"]):
        df = min_reinforcement_costs(df)

    if not split_to_geojson:
        out_file = ('combo_{}'.format(files[0])
                    if out_file is None else out_file)
        out_file = os.path.join(out_dir, out_file)
        logger.info('Saving to {}'.format(out_file))
        if "gpkg" in out_file:
            df.to_file(out_file, driver="GPKG")
        else:
            df.to_csv(out_file, index=False)
        return

    # Split out put in to GeoJSON by POI name
    for poi in set(df['POI Name']):
        out_file = os.path.join(out_dir,
                                "{}_paths.geojson"
                                .format(poi.replace(' ', '_')))
        paths = df[df['POI Name'] == poi].to_crs(epsg=4326)
        logger.info('Writing {} paths for {} to {}'
                    .format(len(paths), poi, out_file))
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
@click.option('--out_file', '-of', default=None, type=STR,
              help='Name for output GeoPackage/CSV file.')
@click.pass_context
def merge_reinforcement_costs(ctx, cost_fpath, reinforcement_cost_fpath,
                              out_file):
    """
    Merge reinforcement costs into transmission costs.
    """
    log_level = "DEBUG" if ctx.obj.get('VERBOSE') else "INFO"
    init_logger('reVX', log_level=log_level)

    costs = (gpd.read_file(cost_fpath)
             if "gpkg" in cost_fpath
             else pd.read_csv(cost_fpath))
    r_costs = (gpd.read_file(reinforcement_cost_fpath)
               if "gpkg" in reinforcement_cost_fpath
               else pd.read_csv(reinforcement_cost_fpath))

    r_costs.index = r_costs.gid

    logger.info("Merging reinforcement costs into transmission costs...")

    r_cols = ["ba_str", "reinforcement_poi_lat", "reinforcement_poi_lon",
              "reinforcement_dist_km", "reinforcement_cost_per_mw"]
    costs[r_cols] = r_costs.loc[costs["trans_gid"], r_cols].values

    logger.info("Writing output to {!r}".format(out_file))

    if "gpkg" in out_file:
        costs.to_file(out_file, driver="GPKG", index=False)
    else:
        costs.to_csv(out_file, index=False)


def get_node_cmd(config, start_index=0):
    """
    Get the node CLI call for Least Cost Xmission

    Parameters
    ----------
    config : reVX.config.least_cost_xmission.LeastCostXmissionConfig
        Least Cost Xmission config object.

    Returns
    -------
    cmd : str
        CLI call to submit to SLURM execution.
    """

    args = ['-n {}'.format(SLURM.s(config.name)),
            'local',
            '-cost {}'.format(SLURM.s(config.cost_fpath)),
            '-feats {}'.format(SLURM.s(config.features_fpath)),
            '-ba {}'.format(SLURM.s(config.balancing_areas_fpath)),
            '-cap {}'.format(SLURM.s(config.capacity_class)),
            '-res {}'.format(SLURM.s(config.resolution)),
            '-xcfg {}'.format(SLURM.s(config.xmission_config)),
            '-mll {}'.format(SLURM.s(config.min_line_length)),
            '-start {}'.format(SLURM.s(start_index)),
            '-step {}'.format(SLURM.s(config.execution_control.nodes or 1)),
            '-nn {}'.format(SLURM.s(config.nn_sinks)),
            '-buffer {}'.format(SLURM.s(config.clipping_buffer)),
            '-bmult {}'.format(SLURM.s(config.barrier_mult)),
            '-mw {}'.format(SLURM.s(config.execution_control.max_workers)),
            '-o {}'.format(SLURM.s(config.dirout)),
            '-log {}'.format(SLURM.s(config.log_directory)),
            ]

    if config.allow_connections_within_states:
        args.append('-acws')
    if config.save_paths:
        args.append('--save_paths')
    if config.radius:
        args.append('-rad {}'.format(config.radius))
    if config.simplify_geo:
        args.append('--simplify-geo {}'.format(config.simplify_geo))

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
               balancing_areas_fpath=config.balancing_areas_fpath,
               capacity_class=config.capacity_class,
               resolution=config.resolution,
               xmission_config=config.xmission_config,
               min_line_length=config.min_line_length,
               sc_point_start_index=0,
               sc_point_step_index=1,
               nn_sinks=config.nn_sinks,
               clipping_buffer=config.clipping_buffer,
               barrier_mult=config.barrier_mult,
               state_connections=config.allow_connections_within_states,
               max_workers=config.execution_control.max_workers,
               out_dir=config.dirout,
               log_dir=config.log_directory,
               verbose=config.log_level,
               radius=config.radius,
               save_paths=config.save_paths,
               simplify_geo=config.simplify_geo,
               )


def eagle(config, start_index=0):
    """
    Run Least Cost Xmission on Eagle HPC.

    Parameters
    ----------
    config : reVX.config.least_cost_xmission.LeastCostXmissionConfig
        Least Cost Xmission config object.
    """
    init_logger('rex', log_level='DEBUG')

    cmd = get_node_cmd(config, start_index)
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
