# -*- coding: utf-8 -*-
# pylint: disable=all
"""
Assembly Areas Command Line Interface
"""
import click
import logging
import os

from rex.utilities.loggers import init_mult
from rex.utilities.cli_dtypes import STR
from rex.utilities.hpc import SLURM
from rex.utilities.utilities import get_class_properties

from reVX.config.offshore import AssemblyAreasConfig
from reVX.offshore.assembly_areas import AssemblyAreas
from reVX import __version__

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--name', '-n', default='AssemblyAreas', type=STR,
              show_default=True,
              help='Job name.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """
    Assembly Areas Command Line Interface
    """
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose
    ctx.obj['NAME'] = name


@main.command()
def valid_config_keys():
    """
    Echo the valid AssemblyArea config keys
    """
    click.echo(', '.join(get_class_properties(AssemblyAreasConfig)))


def run_local(ctx, config):
    """
    Compute Assembly Areas locally using config

    Parameters
    ----------
    ctx : click.ctx
        click ctx object
    config : reVX.config.dist_to_ports.AssemblyAreasConfig
        Assembly Areas config object.
    """
    ctx.obj['NAME'] = config.name
    ctx.invoke(local,
               assembly_areas=config.assembly_areas,
               excl_fpath=config.excl_fpath,
               ports_dset=config.ports_dset,
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
    Compute Assembly Areas from a config.
    """

    config = AssemblyAreasConfig(config)

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
@click.option('--assembly_areas', '-areas', type=click.Path(exists=True),
              required=True,
              help=("Path to csv or json file containing assembly area "
                    "locations. If provided compute distance from ports to "
                    "assembly areas and save as a table to excl_fpath."))
@click.option('--excl_fpath', '-excl', required=True,
              type=click.Path(exists=True),
              help="Filepath to exclusions h5 with techmap dataset.")
@click.option('--ports_dset', '-ports', type=str, show_default=True,
              default='ports_construction_nolimits',
              help="Assembly Areas layer/dataset name in excl_fpath")
@click.option('--log_dir', '-log', default=None, type=STR,
              show_default=True,
              help='Directory to dump log files.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def local(ctx, assembly_areas, excl_fpath, ports_dset, log_dir, verbose):
    """
    Compute Assembly Areas on local hardware
    """
    name = ctx.obj['NAME']
    if 'VERBOSE' in ctx.obj:
        verbose = any((ctx.obj['VERBOSE'], verbose))

    log_modules = [__name__, 'reVX', 'reV', 'rex']
    init_mult(name, log_dir, modules=log_modules, verbose=verbose)

    logger.info('Computing distance from ports to assembly areas in {} \n'
                'Outputs to be stored in: {}'
                .format(assembly_areas, excl_fpath))
    assembly_dset = os.path.basename(assembly_areas).split('.')[0]
    AssemblyAreas.run(assembly_areas, excl_fpath,
                      ports_dset=ports_dset,
                      assembly_dset=assembly_dset)


def get_node_cmd(config):
    """
    Get the node CLI call for Assembly Areas computation.

    Parameters
    ----------
    config : reVX.config.assembly_areas.AssemblyAreasConfig
        Assembly Areas config object.

    Returns
    -------
    cmd : str
        CLI call to submit to SLURM execution.
    """

    args = ['-n {}'.format(SLURM.s(config.name)),
            'local',
            '-areas {}'.format(SLURM.s(config.assembly_areas)),
            '-excl {}'.format(SLURM.s(config.excl_fpath)),
            '-ports {}'.format(SLURM.s(config.ports_dset)),
            '-log {}'.format(SLURM.s(config.log_directory)),
            ]

    if config.log_level == logging.DEBUG:
        args.append('-v')

    cmd = ('python -m reVX.offshore.assembly_areas_cli {}'
           .format(' '.join(args)))
    logger.debug('Submitting the following cli call:\n\t{}'.format(cmd))

    return cmd


def eagle(config):
    """
    Compute Assembly Areas on Eagle HPC.

    Parameters
    ----------
    config : reVX.config.assembly_areas.AssemblyAreasConfig
        Assembly Areas config object.
    """

    cmd = get_node_cmd(config)
    name = config.name
    log_dir = config.log_directory
    stdout_path = os.path.join(log_dir, 'stdout/')

    slurm_manager = SLURM()

    logger.info('Compute Assembly Areas on Eagle with '
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
        msg = ('Kicked off Assembly Areas calculation "{}" '
               '(SLURM jobid #{}) on Eagle.'
               .format(name, out))
    else:
        msg = ('Was unable to kick off Assembly Areas calculation '
               '"{}". Please see the stdout error messages'
               .format(name))

    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running Assembly Areas CLI')
        raise
