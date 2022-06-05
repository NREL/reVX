# -*- coding: utf-8 -*-
# pylint: disable=all
"""
Least Cost Xmission Command Line Interface

TODO - add cmd line doc

"""
import os
import sys
import click
import logging
import warnings
import pandas as pd
import geopandas as gpd

from rex.utilities.loggers import init_mult, create_dirs, init_logger
from rex.utilities.cli_dtypes import STR, INTLIST, INT
from rex.utilities.hpc import SLURM
from rex.utilities.utilities import get_class_properties  # , safe_json_load

from reVX.config.least_cost_xmission import LeastCostXmissionConfig
from reVX.least_cost_xmission.least_cost_xmission import LeastCostXmission
from reVX import __version__

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

    if option != 'eagle':
        raise AttributeError(f'Option {option} is not supported')

    if config.split_gids is None:
        eagle(config)

    # Split gids over mulitple SLURM jobs
    gids = config.sc_point_gids
    name = config.name
    logger.info(f'Splitting {len(gids)} SC points over {config.split_gids} '
                'SLURM jobs')
    for i in range(config.split_gids):
        config.name = f'{name}_{i}'
        config.sc_point_gids = gids[i::config.split_gids]
        eagle(config)


@main.command()
@click.option('--cost_fpath', '-cost', type=click.Path(exists=True),
              required=True,
              help=("Path to h5 file with cost rasters and other required "
                    "layers"))
@click.option('--features_fpath', '-feats', required=True,
              type=click.Path(exists=True),
              help="Path to geopackage with transmission features")
@click.option('--capacity_class', '-cap', type=str, required=True,
              help=("Capacity class of transmission features to connect "
                    "supply curve points to"))
@click.option('--resolution', '-res', type=int,
              show_default=True, default=128,
              help=("SC point resolution"))
@click.option('--xmission_config', '-xcfg', type=STR, show_default=True,
              default=None,
              help=("Path to Xmission config .json"))
@click.option('--sc_point_gids', '-gids', type=INTLIST, show_default=True,
              default=None,
              help=("List of sc_point_gids to connect to"))
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
              help='Save least cost paths and data to geopackage.')
@click.option('--radius', '-rad', type=INT,
              show_default=True, default=None,
              help=("Radius to clip costs raster to in pixels This overrides"
                    "--nn_sinks if set."))
@click.pass_context
def local(ctx, cost_fpath, features_fpath, capacity_class, resolution,
          xmission_config, sc_point_gids, nn_sinks, clipping_buffer,
          barrier_mult, max_workers, out_dir, log_dir, verbose, save_paths,
          radius):
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
    least_costs = LeastCostXmission.run(cost_fpath, features_fpath,
                                        capacity_class,
                                        resolution=resolution,
                                        xmission_config=xmission_config,
                                        sc_point_gids=sc_point_gids,
                                        nn_sinks=nn_sinks,
                                        clipping_buffer=clipping_buffer,
                                        barrier_mult=barrier_mult,
                                        max_workers=max_workers,
                                        save_paths=save_paths,
                                        radius=radius)
    if len(least_costs) == 0:
        logger.error('No paths found.')
        sys.exit()

    ext = 'gpkg' if save_paths else 'csv'
    fn_out = '{}_{}_{}.{}'.format(name, capacity_class, resolution, ext)
    fpath_out = os.path.join(out_dir, fn_out)

    logger.info(f'Writing output to {fpath_out}')
    if save_paths:
        least_costs.to_file(fpath_out, driver='GPKG')
    else:
        least_costs.to_csv(fpath_out, index=False)
    logger.info('Writing output complete')


@main.command()
@click.option('--split-to-geojson', '-s', is_flag=True,
              help='After merging Geopackages, split into GeoJSON by POI name')
@click.option('--out-file', '-of', default=None, type=STR,
              help='Name for output geopackage file')
@click.option('--out-path', '-op', type=click.Path(exists=True),
              default='.', show_default=True,
              help='Output path for output files. Must exist')
@click.argument('gpkg_files', type=click.Path(exists=True), nargs=-1)
@click.pass_context
def merge_output(ctx, split_to_geojson, out_file, out_path, gpkg_files):
    """
    Merge output Geopackage files and optional convert to GeoJSON
    """
    click.echo(f'Loading: {", ".join(gpkg_files)}')
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    gdf = pd.concat([gpd.read_file(f) for f in gpkg_files])
    gdf = gdf.reset_index()
    gdf = gdf[['POI Name', 'State', 'dist_km', 'sc_point_gid', 'geometry']]

    if not split_to_geojson:
        out_file = f'combo_{gpkg_files[0]}' if out_file is None else out_file
        out_file = os.path.join(out_path, out_file)
        click.echo(f'Saving to {out_file}')
        gdf.to_file(out_file, driver="GPKG")
        sys.exit()

    # Split out put in to GeoJSON by POI name
    for poi in set(gdf['POI Name']):
        outf = os.path.join(out_path, f"{poi.replace(' ', '_')}_paths.geojson")
        paths = gdf[gdf['POI Name'] == poi].to_crs(epsg=4326)
        click.echo(f'Writing {len(paths)} paths for {poi} to {outf}')
        paths.to_file(outf, driver="GeoJSON")


def get_node_cmd(config):
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
            '-cap {}'.format(SLURM.s(config.capacity_class)),
            '-res {}'.format(SLURM.s(config.resolution)),
            '-xcfg {}'.format(SLURM.s(config.xmission_config)),
            '-gids {}'.format(SLURM.s(config.sc_point_gids)),
            '-nn {}'.format(SLURM.s(config.nn_sinks)),
            '-buffer {}'.format(SLURM.s(config.clipping_buffer)),
            '-bmult {}'.format(SLURM.s(config.barrier_mult)),
            '-mw {}'.format(SLURM.s(config.execution_control.max_workers)),
            '-o {}'.format(SLURM.s(config.dirout)),
            '-log {}'.format(SLURM.s(config.log_directory)),
            ]

    if config.save_paths:
            args.append('--save_paths')
    if config.radius:
        args.append(f'-rad {config.radius}')

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
               capacity_class=config.capacity_class,
               resolution=config.resolution,
               xmission_config=config.xmission_config,
               sc_point_gids=config.sc_point_gids,
               nn_sinks=config.nn_sinks,
               clipping_buffer=config.clipping_buffer,
               barrier_mult=config.barrier_mult,
               max_workers=config.execution_control.max_workers,
               out_dir=config.dirout,
               log_dir=config.log_directory,
               verbose=config.log_level)


def eagle(config):
    """
    Run Least Cost Xmission on Eagle HPC.

    Parameters
    ----------
    config : reVX.config.least_cost_xmission.LeastCostXmissionConfig
        Least Cost Xmission config object.
    """
    init_logger('rex', log_level='DEBUG')

    cmd = get_node_cmd(config)
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
