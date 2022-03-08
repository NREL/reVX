# -*- coding: utf-8 -*-
"""
Wind Directions Command Line Interface
"""
import click
import logging
import os

from rex.utilities.loggers import init_mult
from rex.utilities.cli_dtypes import STR, INT
from rex.utilities.hpc import SLURM
from rex.utilities.utilities import get_class_properties

from reVX.config.wind_dirs import ProminentWindDirsConfig
from reVX.wind_dirs.prominent_wind_dirs import ProminentWindDirections
from reVX import __version__

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--name', '-n', default='ProminentWindDirs', type=STR,
              show_default=True,
              help='Job name. Default is "ProminentWindDirs".')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """
    Prominent Wind Directions Command Line Interface
    """
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['VERBOSE'] = verbose


@main.command()
def valid_config_keys():
    """
    Echo the valid Prominent Wind Dirs config keys
    """
    click.echo(', '.join(get_class_properties(ProminentWindDirsConfig)))


def run_local(ctx, config):
    """
    Run ProminentWindDirections locally using config

    Parameters
    ----------
    ctx : click.ctx
        click ctx object
    config : reVX.config.prominent_wind_dirs.ProminentWindDirsConfig
        Prominent Wind Directions config object.
    """
    ctx.obj['NAME'] = config.name
    ctx.invoke(local,
               powerrose_h5_fpath=config.powerrose_h5_fpath,
               excl_fpath=config.excl_fpath,
               out_dir=config.dirout,
               agg_dset=config.agg_dset,
               tm_dset=config.tm_dset,
               resolution=config.resolution,
               excl_area=config.excl_area,
               max_workers=config.execution_control.max_workers,
               sites_per_worker=config.execution_control.sites_per_worker,
               log_dir=config.log_directory,
               verbose=config.log_level)


@main.command()
@click.option('--config', '-c', required=True,
              type=click.Path(exists=True),
              help='Filepath to ProminentWindDirections config json file.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config, verbose):
    """
    Run prominent wind directions from a config.
    """

    config = ProminentWindDirsConfig(config)

    if 'VERBOSE' in ctx.obj:
        if any((ctx.obj['VERBOSE'], verbose)):
            config._log_level = logging.DEBUG
    elif verbose:
        config._log_level = logging.DEBUG

    if config.execution_control.option == 'local':
        run_local(ctx, config)

    if config.execution_control.option == 'eagle':
        eagle(config)


@main.command()
@click.option('--powerrose_h5_fpath', '-prh5', required=True,
              type=click.Path(exists=True),
              help="Filepath to .h5 file containing powerrose data")
@click.option('--excl_fpath', '-excl', required=True,
              type=click.Path(exists=True),
              help="Filepath to exclusions h5 with techmap dataset.")
@click.option('--out_dir', '-o', required=True, type=click.Path(),
              help='Directory to dump output files')
@click.option('--agg_dset', '-ad', default='powerrose_100m', type=STR,
              show_default=True,
              help="Powerrose dataset to aggreate")
@click.option('--tm_dset', '-td', default='techmap_wtk', type=STR,
              show_default=True,
              help=("Dataset name in the techmap file containing the "
                    "exclusions-to-resource mapping data,"))
@click.option('--resolution', '-res', default=128, type=INT,
              show_default=True,
              help=("SC resolution, must be input in combination with gid. "
                    "Prefered option is to use the row / col slices to define "
                    "the SC point instead"))
@click.option('--excl_area', '-ea', default=0.0081, type=float,
              show_default=True,
              help="Area of an exclusion cell (square km)")
@click.option('--max_workers', '-mw', default=None, type=INT,
              show_default=True,
              help=("Number of cores to run summary on. None is all "
                    "available cpus"))
@click.option('--sites_per_worker', '-spw', default=1000, type=INT,
              show_default=True,
              help="Number of SC points to process on each parallel worker")
@click.option('--log_dir', '-log', default=None, type=STR,
              show_default=True,
              help='Directory to dump log files. Default is out_dir.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def local(ctx, powerrose_h5_fpath, excl_fpath, out_dir, agg_dset, tm_dset,
          resolution, excl_area, max_workers, sites_per_worker, log_dir,
          verbose):
    """
    Compute prominent wind directions on local hardware
    """
    sites_per_worker = sites_per_worker if sites_per_worker else 1000

    ctx.obj['OUT_DIR'] = out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_fpath = os.path.basename(powerrose_h5_fpath)
    out_fpath = out_fpath.replace('.h5',
                                  '_prominent_dir_{}.csv'
                                  .format(resolution))
    out_fpath = os.path.join(out_dir, out_fpath)

    if log_dir is None:
        log_dir = out_dir

    name = ctx.obj['NAME']
    if 'VERBOSE' in ctx.obj:
        verbose = any((ctx.obj['VERBOSE'], verbose))

    log_modules = [__name__, 'reVX', 'reV', 'rex']
    init_mult(name, log_dir, modules=log_modules, verbose=verbose)

    logger.info('Aggregating Prominent Wind Directions \n'
                'Outputs to be stored in: {}'.format(out_dir))

    ProminentWindDirections.run(powerrose_h5_fpath, excl_fpath,
                                agg_dset=agg_dset, tm_dset=tm_dset,
                                resolution=resolution, excl_area=excl_area,
                                max_workers=max_workers,
                                sites_per_worker=sites_per_worker,
                                out_fpath=out_fpath)


def get_node_cmd(config):
    """
    Get the node CLI call for prominent wind direction computation.

    Parameters
    ----------
    config : reVX.config.prominent_wind_dirs.ProminentWindDirsConfig
        Prominent Wind Directions config object.

    Returns
    -------
    cmd : str
        CLI call to submit to SLURM execution.
    """
    spw = config.execution_control.sites_per_worker
    args = ['-n {}'.format(SLURM.s(config.name)),
            'local',
            '-prh5 {}'.format(SLURM.s(config.powerrose_h5_fpath)),
            '-excl {}'.format(SLURM.s(config.excl_fpath)),
            '-o {}'.format(SLURM.s(config.dirout)),
            '-ad {}'.format(SLURM.s(config.agg_dset)),
            '-td {}'.format(SLURM.s(config.tm_dset)),
            '-res {}'.format(SLURM.s(config.resolution)),
            '-ea {}'.format(SLURM.s(config.excl_area)),
            '-mw {}'.format(SLURM.s(config.execution_control.max_workers)),
            '-spw {}'.format(SLURM.s(spw)),
            '-log {}'.format(SLURM.s(config.log_directory)),
            ]

    if config.log_level == logging.DEBUG:
        args.append('-v')

    cmd = ('python -m reVX.wind_dirs.prominent_wind_dirs_cli {}'
           .format(' '.join(args)))
    logger.debug('Submitting the following cli call:\n\t{}'.format(cmd))

    return cmd


def eagle(config):
    """
    Run prominent wind directions on Eagle HPC.

    Parameters
    ----------
    config : reVX.config.prominent_wind_dirs.ProminentWindDirsConfig
        Prominent Wind Directions config object.
    """

    cmd = get_node_cmd(config)
    name = config.name
    log_dir = config.log_directory
    stdout_path = os.path.join(log_dir, 'stdout/')

    slurm_manager = SLURM()

    logger.info('Running prominent wind directions computation on Eagle with '
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
        msg = ('Kicked off prominent wind direction calculation "{}" '
               '(SLURM jobid #{}) on Eagle.'
               .format(name, out))
    else:
        msg = ('Was unable to kick off prominent wind direction calculation '
               '"{}". Please see the stdout error messages'
               .format(name))

    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running Prominent Wind Directions CLI')
        raise
