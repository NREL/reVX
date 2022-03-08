# -*- coding: utf-8 -*-
# pylint: disable=all
"""
Offshore Inputs Command Line Interface
"""
import click
import logging
import os

from rex.utilities.loggers import init_mult
from rex.utilities.cli_dtypes import STR, STR_OR_LIST
from rex.utilities.hpc import SLURM
from rex.utilities.utilities import get_class_properties, safe_json_load

from reVX.config.offshore import OffshoreInputsConfig
from reVX.offshore.offshore_inputs import OffshoreInputs
from reVX import __version__

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--name', '-n', default='OffshoreInputs', type=STR,
              show_default=True,
              help='Job name.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """
    Offshore Inputs Command Line Interface
    """
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose
    ctx.obj['NAME'] = name


@main.command()
def valid_config_keys():
    """
    Echo the valid Offshore Inputs config keys
    """
    click.echo(', '.join(get_class_properties(OffshoreInputsConfig)))


def run_local(ctx, config):
    """
    Exract offshore inputs locally using config

    Parameters
    ----------
    ctx : click.ctx
        click ctx object
    config : reVX.config.offshore_inputs.OffshoreInputsConfig
        Offshore inputs config object.
    """
    ctx.obj['NAME'] = config.name
    ctx.invoke(local,
               inputs_fpath=config.inputs_fpath,
               offshore_sites=config.offshore_sites,
               input_layers=config.input_layers,
               out_dir=config.dirout,
               tm_dset=config.tm_dset,
               log_dir=config.log_directory,
               verbose=config.log_level)


@main.command()
@click.option('--config', '-c', required=True,
              type=click.Path(exists=True),
              help='Filepath to OffshoreInputs config json file.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config, verbose):
    """
    Extract offshore inputs from a config.
    """

    config = OffshoreInputsConfig(config)

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
@click.option('--inputs_fpath', '-h5', required=True,
              type=STR_OR_LIST,
              help=("Path to offshore inputs .h5 file(s)"))
@click.option('--offshore_sites', '-sites', required=True,
              type=click.Path(exists=True),
              help=("- Path to .csv | .json file with offshore sites meta data"
                    "\n- Path to a WIND Toolkit .h5 file to source site meta"
                    "\n- List, tuple, or vector of offshore gids"
                    "\n- Pre - extracted site meta DataFrame"))
@click.option('--input_layers', '-layers', type=click.Path(exists=True),
              required=True,
              help=("Path to json file containing input layer, list of input "
                    "layers, to extract, or dictionary mapping the input "
                    "layers to extract to the column names to save them under"
                    ))
@click.option('--out_dir', '-o', required=True, type=click.Path(),
              help='Directory to dump output files')
@click.option('--tm_dset', '-tm', default='techmap_wtk', type=str,
              show_default=True,
              help=("Dataset / layer name for wind toolkit techmap"))
@click.option('--log_dir', '-log', default=None, type=STR,
              show_default=True,
              help='Directory to dump log files.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def local(ctx, inputs_fpath, offshore_sites, input_layers, out_dir, tm_dset,
          log_dir, verbose):
    """
    Extract offshore inputs on local hardware
    """
    name = ctx.obj['NAME']
    if 'VERBOSE' in ctx.obj:
        verbose = any((ctx.obj['VERBOSE'], verbose))

    log_modules = [__name__, 'reVX', 'reV', 'rex']
    init_mult(name, log_dir, modules=log_modules, verbose=verbose)

    out_fpath = os.path.basename(input_layers).replace('.json', '.csv')
    out_fpath = os.path.join(out_dir, out_fpath)

    input_layers = safe_json_load(input_layers)['input_layers']

    logger.info('Extracting {} from {} and saving to {}'
                .format(input_layers, inputs_fpath, out_fpath))
    OffshoreInputs.extract(inputs_fpath, offshore_sites, tm_dset=tm_dset,
                           input_layers=input_layers, out_fpath=out_fpath)


def get_node_cmd(config):
    """
    Get the node CLI call to extract offshore inputs

    Parameters
    ----------
    config : reVX.config.offshore_inputs.OffshoreInputConfig
        Offshore Inputs config object.

    Returns
    -------
    cmd : str
        CLI call to submit to SLURM execution.
    """

    args = ['-n {}'.format(SLURM.s(config.name)),
            'local',
            '-h5 {}'.format(SLURM.s(config.inputs_fpath)),
            '-sites {}'.format(SLURM.s(config.offshore_sites)),
            '-layers {}'.format(SLURM.s(config.input_layers)),
            '-o {}'.format(SLURM.s(config.dirout)),
            '-tm {}'.format(SLURM.s(config.tm_dset)),
            '-log {}'.format(SLURM.s(config.log_directory)),
            ]

    if config.log_level == logging.DEBUG:
        args.append('-v')

    cmd = ('python -m reVX.offshore.inputs_cli {}'
           .format(' '.join(args)))
    logger.debug('Submitting the following cli call:\n\t{}'.format(cmd))

    return cmd


def eagle(config):
    """
    extract offshore inputs on Eagle HPC.

    Parameters
    ----------
    config : reVX.config.offshore_inputs.OffshoreInputConfig
        Offshore Inputs config object.
    """

    cmd = get_node_cmd(config)
    name = config.name
    log_dir = config.log_directory
    stdout_path = os.path.join(log_dir, 'stdout/')

    slurm_manager = SLURM()

    logger.info('Extracting offshore inputs on Eagle with '
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
        msg = ('Kicked off offshore inputs extraction "{}" '
               '(SLURM jobid #{}) on Eagle.'
               .format(name, out))
    else:
        msg = ('Was unable to kick off offshore inputs extraction '
               '"{}". Please see the stdout error messages'
               .format(name))

    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running offshore inputs CLI')
        raise
