# -*- coding: utf-8 -*-
# pylint: disable=all
"""
Xmission Cost Creator Command Line Interface
"""
import click
import logging
import os

from rex.utilities.loggers import init_mult
from rex.utilities.cli_dtypes import STR
from rex.utilities.hpc import SLURM
from rex.utilities.utilities import get_class_properties, safe_json_load

from reVX.config.least_cost_xmission import CostCreatorConfig
from reVX.least_cost_xmission.cost_creator import XmissionCostCreator
from reVX import __version__

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--name', '-n', default='CostCreator', type=STR,
              show_default=True,
              help='Job name.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """
    Least Cost Xmission Cost Creator Command Line Interface
    """
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose
    ctx.obj['NAME'] = name


@main.command()
def valid_config_keys():
    """
    Echo the valid Least Cost Xmission Cost Creator config keys
    """
    click.echo(', '.join(get_class_properties(CostCreatorConfig)))


def run_local(ctx, config):
    """
    Run Least Cost Xmission Cost Creator locally using config

    Parameters
    ----------
    ctx : click.ctx
        click ctx object
    config : reVX.config.least_cost_xmission.CostCreatorConfig
        Cost Creator config object.
    """
    ctx.obj['NAME'] = config.name
    ctx.invoke(local,
               h5_fpath=config.h5_fpath,
               iso_regions=config.iso_regions,
               excl_h5=config.excl_h5,
               cost_configs=config.cost_configs,
               slope_layer=config.slope_layer,
               nlcd_layer=config.nlcd_layer,
               default_mults=config.default_mults,
               tiff_dir=config.tiff_dir,
               extra_layers=config.extra_layers,
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
    Run Least Cost Xmission Cost Creator from a config.
    """

    config = CostCreatorConfig(config)

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
@click.option('--h5_fpath', '-h5', type=click.Path(),
              required=True,
              help=("H5 file to save costs to"))
@click.option('--iso_regions', '-iso', required=True,
              type=click.Path(exists=True),
              help="File with raster of ISO regions.")
@click.option('--excl_h5', '-excl', type=STR,
              show_default=True, default=None,
              help=("Path to exclusion .h5 file containing NLCD and "
                    "slope layers, if None use h5_fpath if None assume "
                    "NLCD and slope layers are in self._excl_h5"))
@click.option('--cost_configs', '-ccfg', type=STR,
              show_default=True, default=None,
              help=("JSON file with cost configs"))
@click.option('--slope_layer', '-slope', type=str, show_default=True,
              default="srtm_slope",
              help=("Name of slope layer in excl_h5"))
@click.option('--nlcd_layer', '-nlcd', type=str, show_default=True,
              default="usa_mrlc_nlcd2011",
              help=("Name of NLCD (land use) layer in excl_h5"))
@click.option('--default_mults', '-dm', type=STR,
              show_default=True, default=None,
              help=("JSON of Multipliers for regions not specified in "
                    "iso_mults_fpath"))
@click.option('--tiff_dir', '-tiff', type=STR,
              show_default=True, default=None,
              help=("Path to save costs and intermediary rasters as geotiffs"))
@click.option('--extra_layers', '-lyrs', type=STR,
              show_default=True, default=None,
              help=("JSON with Extra layers to add to h5 file, for example "
                    "dist_to_coast"))
@click.option('--log_dir', '-log', default=None, type=STR,
              show_default=True,
              help='Directory to dump log files.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def local(ctx, h5_fpath, iso_regions, excl_h5, cost_configs, slope_layer,
          nlcd_layer, default_mults, tiff_dir, extra_layers, log_dir, verbose):
    """
    Run Least Cost Xmission Cost Creator on local hardware
    """
    name = ctx.obj['NAME']
    if 'VERBOSE' in ctx.obj:
        verbose = any((ctx.obj['VERBOSE'], verbose))

    log_modules = [__name__, 'reVX', 'reV', 'rex']
    init_mult(name, log_dir, modules=log_modules, verbose=verbose)

    if isinstance(default_mults, str):
        default_mults = safe_json_load(default_mults)

    if isinstance(extra_layers, str):
        extra_layers = safe_json_load(extra_layers)

    logger.info('Computing Xmission Cost layers and writing them to {}'
                .format(h5_fpath))
    XmissionCostCreator.run(h5_fpath, iso_regions, excl_h5=excl_h5,
                            cost_configs=cost_configs,
                            slope_layer=slope_layer, nlcd_layer=nlcd_layer,
                            default_mults=default_mults, tiff_dir=tiff_dir,
                            extra_layers=extra_layers)


def get_node_cmd(config):
    """
    Get the node CLI call for Least Cost Xmission Cost Creator

    Parameters
    ----------
    config : reVX.config.least_cost_xmission.CostCreatorConfig
        Cost Creator config object.

    Returns
    -------
    cmd : str
        CLI call to submit to SLURM execution.
    """

    args = ['-n {}'.format(SLURM.s(config.name)),
            'local',
            '-h5 {}'.format(SLURM.s(config.h5_fpath)),
            '-iso {}'.format(SLURM.s(config.iso_regions)),
            '-excl {}'.format(SLURM.s(config.excl_h5)),
            '-ccfg {}'.format(SLURM.s(config.cost_configs)),
            '-slope {}'.format(SLURM.s(config.slope_layer)),
            '-nlcd {}'.format(SLURM.s(config.nlcd_layer)),
            '-dm {}'.format(SLURM.s(config.default_mults)),
            '-tiff {}'.format(SLURM.s(config.tiff_dir)),
            '-lyrs {}'.format(SLURM.s(config.extra_layers)),
            '-log {}'.format(SLURM.s(config.log_directory)),
            ]

    if config.log_level == logging.DEBUG:
        args.append('-v')

    cmd = ('python -m reVX.least_cost_xmission.cost_creator_cli {}'
           .format(' '.join(args)))
    logger.debug('Submitting the following cli call:\n\t{}'.format(cmd))

    return cmd


def eagle(config):
    """
    Run Least Cost Xmission Cost Creator on Eagle HPC.

    Parameters
    ----------
    config : reVX.config.least_cost_xmission.CostCreatorConfig
        Cost Creator config object.
    """

    cmd = get_node_cmd(config)
    name = config.name
    log_dir = config.log_directory
    stdout_path = os.path.join(log_dir, 'stdout/')

    slurm_manager = SLURM()

    logger.info('Running Least Cost Xmission Cost Creator on Eagle with '
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
        msg = ('Kicked off Least Cost Xmission Cost Creator "{}" '
               '(SLURM jobid #{}) on Eagle.'
               .format(name, out))
    else:
        msg = ('Was unable to kick off Least Cost Xmission Cost Creator '
               '"{}". Please see the stdout error messages'
               .format(name))

    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running Least Cost Xmission Cost Creator CLI')
        raise
