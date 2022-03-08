# -*- coding: utf-8 -*-
# pylint: disable=all
"""
Least Cost Xmission Command Line Interface
"""
import click
import logging
import os

from rex.utilities.loggers import init_mult, create_dirs
from rex.utilities.cli_dtypes import STR, INT
from rex.utilities.hpc import SLURM
from rex.utilities.utilities import get_class_properties

from reVX.config.least_cost_xmission import LeastCostPathsConfig
from reVX.least_cost_xmission.config import XmissionConfig
from reVX.least_cost_xmission.least_cost_paths import LeastCostPaths
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
               capacity_class=config.capacity_class,
               xmission_config=config.xmission_config,
               barrier_mult=config.barrier_mult,
               max_workers=config.execution_control.max_workers,
               save_paths=config.save_paths,
               out_dir=config.dirout,
               log_dir=config.log_directory,
               verbose=config.log_level)


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

    if config.execution_control.option == 'eagle':
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
@click.option('--xmission_config', '-xcfg', type=STR, show_default=True,
              default=None,
              help=("Path to Xmission config .json"))
@click.option('--barrier_mult', '-bmult', type=float,
              show_default=True, default=100,
              help=("Tranmission barrier multiplier, used when computing the "
                    "least cost tie-line path"))
@click.option('--max_workers', '-mw', type=INT,
              show_default=True, default=None,
              help=("Number of workers to use for processing, if 1 run in "
                    "serial, if None use all available cores"))
@click.option('--save_paths', '-paths', is_flag=True,
              help="Flag to save least cost path as a multi-line geometry")
@click.option('--out_dir', '-o', type=STR, default='./',
              show_default=True,
              help='Directory to save least cost Paths values to.')
@click.option('--log_dir', '-log', default=None, type=STR,
              show_default=True,
              help='Directory to dump log files.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def local(ctx, cost_fpath, features_fpath, capacity_class, xmission_config,
          barrier_mult, max_workers, save_paths, out_dir, log_dir, verbose):
    """
    Run Least Cost Paths on local hardware
    """
    name = ctx.obj['NAME']
    if 'VERBOSE' in ctx.obj:
        verbose = any((ctx.obj['VERBOSE'], verbose))

    log_modules = [__name__, 'reVX', 'reV', 'rex']
    init_mult(name, log_dir, modules=log_modules, verbose=verbose)

    create_dirs(out_dir)
    logger.info('Computing Least Cost Paths connections and writing them {}'
                .format(out_dir))
    xmission_config = XmissionConfig(config=xmission_config)
    least_costs = LeastCostPaths.run(cost_fpath, features_fpath,
                                     capacity_class,
                                     xmission_config=xmission_config,
                                     barrier_mult=barrier_mult,
                                     max_workers=max_workers,
                                     save_paths=save_paths)

    capacity_class = xmission_config._parse_cap_class(capacity_class)
    cap = xmission_config['power_classes'][capacity_class]
    kv = xmission_config.capacity_to_kv(capacity_class)
    fn_out = '{}_{}MW_{}kV'.format(name, cap, kv)
    fpath_out = os.path.join(out_dir, fn_out)
    if save_paths:
        fpath_out += '.gpkg'
        least_costs.to_file(fpath_out, layer='paths', driver="GPKG")
    else:
        fpath_out += '.csv'
        least_costs.to_csv(fpath_out, index=False)


def get_node_cmd(config):
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
            '-cap {}'.format(SLURM.s(config.capacity_class)),
            '-bmult {}'.format(SLURM.s(config.barrier_mult)),
            '-mw {}'.format(SLURM.s(config.execution_control.max_workers)),
            '-o {}'.format(SLURM.s(config.dirout)),
            '-log {}'.format(SLURM.s(config.log_directory)),
            ]

    if config.save_paths:
        args.append('-paths')

    if config.log_level == logging.DEBUG:
        args.append('-v')

    cmd = ('python -m reVX.least_cost_xmission.least_cost_paths_cli {}'
           .format(' '.join(args)))
    logger.debug('Submitting the following cli call:\n\t{}'.format(cmd))

    return cmd


def eagle(config):
    """
    Run Least Cost Paths on Eagle HPC.

    Parameters
    ----------
    config : reVX.config.least_cost_xmission.LeastCostPathsConfig
        Least Cost Paths config object.
    """

    cmd = get_node_cmd(config)
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
