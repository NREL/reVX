# -*- coding: utf-8 -*-
"""
Wind Directions Command Line Interface
"""
import click
import logging
import os

from rex.utilities.loggers import init_mult
from rex.utilities.cli_dtypes import STR, INT, STRLIST, FLOAT
from rex.utilities.hpc import SLURM
from rex.utilities.utilities import get_class_properties

from reVX.config.wind_dirs import MeanWindDirsConfig
from reVX.wind_dirs.mean_wind_dirs import MeanWindDirections
from reVX import __version__

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--name', '-n', default='MeanWindDirs', type=STR,
              show_default=True,
              help='Job name.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """
    Mean Wind Directions Command Line Interface
    """
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['VERBOSE'] = verbose


@main.command()
def valid_config_keys():
    """
    Echo the valid Mean Wind Dirs config keys
    """
    click.echo(', '.join(get_class_properties(MeanWindDirsConfig)))


def run_local(ctx, config):
    """
    Run MeanWindDirections locally using config

    Parameters
    ----------
    ctx : click.ctx
        click ctx object
    config : reVX.config.mean_wind_dirs.MeanWindDirsConfig
        Mean Wind Directions config object.
    """
    ctx.obj['NAME'] = config.name
    ctx.invoke(local,
               res_h5_fpath=config.res_h5_fpath,
               excl_fpath=config.excl_fpath,
               wdir_dsets=config.wdir_dsets,
               out_dir=config.dirout,
               tm_dset=config.tm_dset,
               excl_dict=config.excl_dict,
               resolution=config.resolution,
               excl_area=config.excl_area,
               area_filter_kernel=config.area_filter_kernel,
               min_area=config.min_area,
               max_workers=config.execution_control.max_workers,
               sites_per_worker=config.execution_control.sites_per_worker,
               log_dir=config.log_directory,
               verbose=config.log_level)


@main.command()
@click.option('--config', '-c', required=True,
              type=click.Path(exists=True),
              help='Filepath to MeanWindDirections config json file.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config, verbose):
    """
    Run mean wind directions from a config.
    """

    config = MeanWindDirsConfig(config)

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
@click.option('--res_h5_fpath', '-res', required=True,
              type=click.Path(exists=True),
              help="Filepath to .h5 file containing wind direction data")
@click.option('--excl_fpath', '-excl', required=True,
              type=click.Path(exists=True),
              help="Filepath to exclusions h5 with techmap dataset.")
@click.option('--wdir_dsets', '-dsets', required=True, type=STRLIST,
              help="Wind direction dataset to average")
@click.option('--out_dir', '-o', required=True, type=click.Path(),
              help='Directory to dump output files')
@click.option('--tm_dset', '-td', default='techmap_wtk', type=STR,
              show_default=True,
              help=("Dataset name in the techmap file containing the "
                    "exclusions-to-resource mapping data,"))
@click.option('--excl_dict', '-exd', type=STR, default=None,
              help=('String representation of a dictionary of exclusion '
                    'LayerMask arguments {layer: {kwarg: value}} where layer '
                    'is a dataset in excl_fpath and kwarg can be '
                    '"inclusion_range", "exclude_values", "include_values", '
                    '"inclusion_weights", "force_inclusion_values", '
                    '"use_as_weights", "exclude_nodata", and/or "weight".'))
@click.option('--resolution', '-res', default=128, type=INT,
              show_default=True,
              help=("SC resolution, must be input in combination with gid. "
                    "Prefered option is to use the row / col slices to define "
                    "the SC point instead"))
@click.option('--excl_area', '-ea', default=None, type=float,
              show_default=True,
              help="Area of an exclusion cell (square km)")
@click.option('--area_filter_kernel', '-afk', type=STR, default='queen',
              help='Contiguous area filter kernel name ("queen", "rook").')
@click.option('--min_area', '-ma', type=FLOAT, default=None,
              help='Contiguous area filter minimum area, default is None '
              '(No minimum area filter).')
@click.option('--sites_per_worker', '-spw', default=1000, type=INT,
              show_default=True,
              help="Number of SC points to process on each parallel worker")
@click.option('--max_workers', '-mw', default=None, type=INT,
              show_default=True,
              help=("Number of cores to run summary on. None is all "
                    "available cpus"))
@click.option('--log_dir', '-log', default=None, type=STR,
              show_default=True,
              help='Directory to dump log files. Default is out_dir.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def local(ctx, res_h5_fpath, excl_fpath, wdir_dsets, out_dir, tm_dset,
          excl_dict, resolution, excl_area, area_filter_kernel, min_area,
          sites_per_worker, max_workers, log_dir, verbose):
    """
    Compute mean wind directions on local hardware
    """
    sites_per_worker = sites_per_worker if sites_per_worker else 1000

    ctx.obj['OUT_DIR'] = out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_fpath = os.path.basename(res_h5_fpath)
    out_fpath = out_fpath.replace('.h5',
                                  '_means_{}.h5'
                                  .format(resolution))
    out_fpath = os.path.join(out_dir, out_fpath)

    if log_dir is None:
        log_dir = out_dir

    name = ctx.obj['NAME']
    if 'VERBOSE' in ctx.obj:
        verbose = any((ctx.obj['VERBOSE'], verbose))

    log_modules = [__name__, 'reVX', 'reV', 'rex']
    init_mult(name, log_dir, modules=log_modules, verbose=verbose)

    logger.info('Averaging Wind Directions \n'
                'Outputs to be stored in: {}'.format(out_dir))

    MeanWindDirections.run(res_h5_fpath, excl_fpath, wdir_dsets,
                           tm_dset=tm_dset,
                           excl_dict=excl_dict,
                           area_filter_kernel=area_filter_kernel,
                           min_area=min_area,
                           resolution=resolution,
                           excl_area=excl_area,
                           max_workers=max_workers,
                           sites_per_worker=sites_per_worker,
                           out_fpath=out_fpath)


def get_node_cmd(config):
    """
    Get the node CLI call for mean wind direction computation.

    Parameters
    ----------
    config : reVX.config.mean_wind_dirs.MeanWindDirsConfig
        Mean Wind Directions config object.

    Returns
    -------
    cmd : str
        CLI call to submit to SLURM execution.
    """
    spw = config.execution_control.sites_per_worker
    args = ['-n {}'.format(SLURM.s(config.name)),
            'local',
            '-res {}'.format(SLURM.s(config.res_h5_fpath)),
            '-excl {}'.format(SLURM.s(config.excl_fpath)),
            '-dsets {}'.format(SLURM.s(config.wdir_dsets)),
            '-o {}'.format(SLURM.s(config.dirout)),
            '-td {}'.format(SLURM.s(config.tm_dset)),
            '-exd {}'.format(SLURM.s(config.excl_dict)),
            '-res {}'.format(SLURM.s(config.resolution)),
            '-ea {}'.format(SLURM.s(config.excl_area)),
            '-afk {}'.format(SLURM.s(config.area_filter_kernel)),
            '-ma {}'.format(SLURM.s(config.min_area)),
            '-mw {}'.format(SLURM.s(config.execution_control.max_workers)),
            '-spw {}'.format(SLURM.s(spw)),
            '-log {}'.format(SLURM.s(config.log_directory)),
            ]

    if config.log_level == logging.DEBUG:
        args.append('-v')

    cmd = ('python -m reVX.wind_dirs.mean_wind_dirs_cli {}'
           .format(' '.join(args)))
    logger.debug('Submitting the following cli call:\n\t{}'.format(cmd))

    return cmd


def eagle(config):
    """
    Run mean wind directions on Eagle HPC.

    Parameters
    ----------
    config : reVX.config.mean_wind_dirs.MeanWindDirsConfig
        Mean Wind Directions config object.
    """

    cmd = get_node_cmd(config)
    name = config.name
    log_dir = config.log_directory
    stdout_path = os.path.join(log_dir, 'stdout/')

    slurm_manager = SLURM()

    logger.info('Averaging wind directions on Eagle with '
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
        msg = ('Kicked off mean wind direction calculation "{}" '
               '(SLURM jobid #{}) on Eagle.'
               .format(name, out))
    else:
        msg = ('Was unable to kick off mean wind direction calculation '
               '"{}". Please see the stdout error messages'
               .format(name))

    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running Mean Wind Directions CLI')
        raise
