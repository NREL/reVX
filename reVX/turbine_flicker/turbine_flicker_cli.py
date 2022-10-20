# -*- coding: utf-8 -*-
"""
Turbine Flicker Command Line Interface
"""
import click
import logging
import os

from rex.utilities.loggers import init_mult
from rex.utilities.cli_dtypes import STR, INT, STRFLOAT
from rex.utilities.hpc import SLURM
from rex.utilities.utilities import get_class_properties

from reVX.config.turbine_flicker import TurbineFlickerConfig
from reVX.turbine_flicker.turbine_flicker import TurbineFlicker
from reVX.turbine_flicker.regulations import FlickerRegulations
from reVX import __version__

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--name', '-n', default='TurbineFlicker', type=STR,
              show_default=True,
              help='Job name.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """
    Turbine Flicker Command Line Interface
    """
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['VERBOSE'] = verbose


@main.command()
def valid_config_keys():
    """
    Echo the valid Turbine Flicker config keys
    """
    click.echo(', '.join(get_class_properties(TurbineFlickerConfig)))


def run_local(ctx, config):
    """
    Run Turbine Flicker locally using config

    Parameters
    ----------
    ctx : click.ctx
        click ctx object
    config : reVX.config.turbine_flicker.TurbineFlickerConfig
        turbine flicker config object.
    """
    ctx.obj['NAME'] = config.name
    ctx.invoke(local,
               excl_fpath=config.excl_fpath,
               res_fpath=config.res_fpath,
               building_layer=config.building_layer,
               hub_height=config.hub_height,
               rotor_diameter=config.rotor_diameter,
               out_layer=config.out_layer,
               out_dir=config.dirout,
               tm_dset=config.tm_dset,
               building_threshold=config.building_threshold,
               flicker_threshold=config.flicker_threshold,
               resolution=config.resolution,
               grid_cell_size=config.grid_cell_size,
               max_flicker_exclusion_range=config.max_flicker_exclusion_range,
               regs_fpath=config.regs_fpath,
               max_workers=config.execution_control.max_workers,
               replace=config.replace,
               hsds=config.hsds,
               log_dir=config.log_directory,
               verbose=config.log_level)


@main.command()
@click.option('--config', '-c', required=True,
              type=click.Path(exists=True),
              help='Filepath to TurbineFlicker config json file.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config, verbose):
    """
    Run turbine flicker from a config.
    """

    config = TurbineFlickerConfig(config)

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
@click.option('--excl_fpath', '-excl', required=True,
              type=click.Path(exists=True),
              help="Filepath to exclusions h5 with techmap dataset.")
@click.option('--res_fpath', '-ref', required=True,
              type=click.Path(exists=True),
              help="Filepath to .h5 file containing wind direction data")
@click.option('--building_layer', '-bldl', type=str,
              help=('Exclusion layer containing buildings from which turbine '
                    'flicker exclusions will be computed.'))
@click.option('--hub_height', '-h', required=True, type=int,
              help=('Hub-height in meters to compute turbine shadow flicker.'))
@click.option('--rotor_diameter', '-rd', required=True, type=int,
              help=('Rotor diameter in meters to compute turbine shadow '
                    'flicker.'))
@click.option('--out_layer', '-ol', default=None, type=STR,
              show_default=True,
              help=("Layer to save exclusions under. Layer will be saved in "
                    "excl_fpath, if not provided will be generated from the "
                    "building_layer name and hub-height"))
@click.option('--out_dir', '-o', required=True, type=STR,
              help=('Directory to save setbacks geotiff(s) into'))
@click.option('--tm_dset', '-td', default='techmap_wtk', type=STR,
              show_default=True,
              help=("Dataset name in the techmap file containing the "
                    "exclusions-to-resource mapping data"))
@click.option('--building_threshold', '-bldt', default=0, type=float,
              show_default=True,
              help=("Threshold for exclusion layer values to identify pixels "
                    "with buildings. Values are % of pixels containing a "
                    "building"))
@click.option('--flicker_threshold', '-ft', default=30, type=float,
              show_default=True,
              help=("Maximum number of allowable flicker hours"))
@click.option('--resolution', '-res', default=128, type=INT,
              show_default=True,
              help=("SC resolution, must be input in combination with gid. "
                    "Prefered option is to use the row / col slices to define "
                    "the SC point instead"))
@click.option('--grid_cell_size', '-gcs', default=90, type=INT,
              show_default=True,
              help=("Length (m) of a side of each grid cell in `excl_fpath`."))
@click.option('--max_flicker_exclusion_range', '-mfer', default="10x",
              type=STRFLOAT, show_default=True,
              help=("Max distance (m) that flicker exclusions will extend in "
                    "any of the cardinal directions. Can also be a string "
                    "like ``'10x'`` (default), which is interpreted as 10 "
                    "times the turbine rotor diameter. Note that increasing "
                    "this value can lead to drastically instead memory "
                    "requirements. This value may be increased slightly in "
                    "order to yield odd exclusion array shapes."))
@click.option('--regs_fpath', '-regs', default=None, type=STR,
              show_default=True,
              help=('Path to regulations .csv file, if None create '
                    'generic setbacks using max - tip height * "multiplier", '
                    'by default None'))
@click.option('--max_workers', '-mw', default=None, type=INT,
              show_default=True,
              help=("Number of cores to run summary on. None is all "
                    "available cpus"))
@click.option('--replace', '-r', is_flag=True,
              help=('Flag to replace local layer data with arr if layer '
                    'already exists in the exclusion .h5 file'))
@click.option('--hsds', '-hsds', is_flag=True,
              help=('Flag to use h5pyd to handle .h5 domain hosted on AWS '
                    'behind HSDS'))
@click.option('--log_dir', '-log', default=None, type=STR,
              show_default=True,
              help='Directory to dump log files. Default is out_dir.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def local(ctx, excl_fpath, res_fpath, building_layer, hub_height,
          rotor_diameter, out_layer, out_dir, tm_dset, building_threshold,
          flicker_threshold, resolution, grid_cell_size,
          max_flicker_exclusion_range, regs_fpath, max_workers, replace, hsds,
          log_dir, verbose):
    """
    Compute turbine flicker on local hardware
    """
    if out_layer is not None:
        out_layers = {os.path.basename(building_layer): out_layer}
    else:
        out_layers = {}

    name = ctx.obj['NAME']
    if 'VERBOSE' in ctx.obj:
        verbose = any((ctx.obj['VERBOSE'], verbose))

    log_modules = [__name__, 'reVX', 'reV', 'rex']
    init_mult(name, log_dir, modules=log_modules, verbose=verbose)

    logger.info('Computing Turbine Flicker Exclusions from structures in {}'
                .format(building_layer))
    logger.debug('Flicker to be computed with:\n'
                 '- building_layer = {}\n'
                 '- hub_height = {}\n'
                 '- rotor_diameter = {}\n'
                 '- tm_dset = {}\n'
                 '- building_threshold = {}\n'
                 '- flicker_threshold = {}\n'
                 '- resolution = {}\n'
                 '- grid_cell_size = {}\n'
                 '- max_flicker_exclusion_range = {}\n'
                 '- regs_fpath = {}\n'
                 '- using max_workers = {}\n'
                 '- replace layer if needed = {}\n'
                 '- out_layer = {}\n'
                 .format(building_layer, hub_height, rotor_diameter,
                         tm_dset, building_threshold, flicker_threshold,
                         resolution, grid_cell_size,
                         max_flicker_exclusion_range, regs_fpath, max_workers,
                         replace, out_layer))

    regulations = FlickerRegulations(hub_height, rotor_diameter,
                                     flicker_threshold, regs_fpath)
    TurbineFlicker.run(excl_fpath, building_layer, out_dir,
                       res_fpath=res_fpath,
                       regulations=regulations,
                       building_threshold=building_threshold,
                       resolution=resolution,
                       grid_cell_size=grid_cell_size,
                       max_flicker_exclusion_range=max_flicker_exclusion_range,
                       tm_dset=tm_dset, max_workers=max_workers,
                       replace=replace, hsds=hsds, out_layers=out_layers)


def get_node_cmd(config):
    """
    Get the node CLI call for turbine flicker computation.

    Parameters
    ----------
    config : reVX.config.turbine_flicker.TurbineFlickerConfig
        Turbine Flicker config object.

    Returns
    -------
    cmd : str
        CLI call to submit to SLURM execution.
    """
    args = ['-n {}'.format(SLURM.s(config.name)),
            'local',
            '-excl {}'.format(SLURM.s(config.excl_fpath)),
            '-ref {}'.format(SLURM.s(config.res_fpath)),
            '-bldl {}'.format(SLURM.s(config.building_layer)),
            '-h {}'.format(SLURM.s(config.hub_height)),
            '-rd {}'.format(SLURM.s(config.rotor_diameter)),
            '-ol {}'.format(SLURM.s(config.out_layer)),
            '-o {}'.format(SLURM.s(config.dirout)),
            '-td {}'.format(SLURM.s(config.tm_dset)),
            '-bldt {}'.format(SLURM.s(config.building_threshold)),
            '-ft {}'.format(SLURM.s(config.flicker_threshold)),
            '-res {}'.format(SLURM.s(config.resolution)),
            '-gcs {}'.format(SLURM.s(config.grid_cell_size)),
            '-mfer {}'.format(SLURM.s(config.max_flicker_exclusion_range)),
            '-mw {}'.format(SLURM.s(config.execution_control.max_workers)),
            '-regs {}'.format(SLURM.s(config.regs_fpath)),
            '-log {}'.format(SLURM.s(config.log_directory)),
            ]

    if config.replace:
        args.append('-r')

    if config.hsds:
        args.append('-hsds')

    if config.log_level == logging.DEBUG:
        args.append('-v')

    cmd = ('python -m reVX.turbine_flicker.turbine_flicker_cli {}'
           .format(' '.join(args)))
    logger.debug('Submitting the following cli call:\n\t{}'.format(cmd))

    return cmd


def eagle(config):
    """
    Run turbine flicker on Eagle HPC.

    Parameters
    ----------
    config : reVX.config.turbine_flicker.TurbineFlickerConfig
        turbine flicker config object.
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
        msg = ('Kicked off turbine flicker calculation "{}" '
               '(SLURM jobid #{}) on Eagle.'
               .format(name, out))
    else:
        msg = ('Was unable to kick off turbine flicker calculation '
               '"{}". Please see the stdout error messages'
               .format(name))

    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running turbine flicker CLI')
        raise
