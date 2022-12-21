# -*- coding: utf-8 -*-
"""
Setbacks CLI
"""
import click
from copy import deepcopy
import logging
import os
from pathlib import Path

from rex.utilities.loggers import init_mult, init_logger
from rex.utilities.cli_dtypes import STR, FLOAT, INT
from rex.utilities.hpc import SLURM
from rex.utilities.utilities import get_class_properties, dict_str_load

from reVX.config.setbacks import SetbacksConfig
from reVX.setbacks import SETBACKS
from reVX.setbacks.regulations import (validate_setback_regulations_input,
                                       select_setback_regulations)
from reVX.setbacks.setbacks_converter import parse_setbacks
from reVX.handlers.geotiff import Geotiff
from reVX.utilities import ExclusionsConverter
from reVX import __version__

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--name', '-n', default=SetbacksConfig.NAME,
              type=STR, show_default=True,
              help='Job name. Default is {!r}.'.format(SetbacksConfig.NAME))
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """
    Setbacks Command Line Interface
    """
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['VERBOSE'] = verbose


@main.command()
def valid_config_keys():
    """
    Echo the valid Setbacks config keys
    """
    click.echo(', '.join(get_class_properties(SetbacksConfig)))


@main.command()
@click.option('--tiff_dir', '-td', required=True,
              type=click.Path(exists=True),
              help=("Path to directory containing geotiffs to be merged."))
@click.option('--out_file', '-o', required=True,
              type=click.Path(),
              help=("Path to output tiff file."))
@click.option('--are_partial_inclusions', '-inclusions', is_flag=True,
              help=('Flag to indicate that the data in the tiff layers '
                    'represent partial inclusion values (i.e. 0.25 = 25% '
                    'included), NOT typical exclusion values (i.e. '
                    '1 = exclude pixel)'))
@click.pass_context
def merge(ctx, tiff_dir, out_file, are_partial_inclusions):
    """
    Combine setbacks geotiffs into a single exclusion (or inclusion) layer.

    This command assumes the data in separate files is non-overlapping.
    In other words, a file containing setbacks exclusions for Illinois
    should not contain any exclusions for Indiana, assuming the setbacks
    for Indiana are in a separate tif file in the same directory.
    """
    log_level = "DEBUG" if ctx.obj.get('VERBOSE') else "INFO"
    init_logger('reVX', log_level=log_level)

    logger.info("Merging tiff files in {!r}".format(tiff_dir))
    setbacks = [path.as_posix() for path in Path(tiff_dir).glob("*.tif*")]
    if not setbacks:
        msg = ("Did not find any files ending in '.tif' in directory: {}"
               .format(tiff_dir))
        logger.error(msg)
        raise FileNotFoundError(msg)

    with Geotiff(setbacks[0]) as tif:
        profile = tif.profile

    out_file = Path(out_file).resolve().as_posix()
    combined_setbacks = parse_setbacks(
        setbacks, is_inclusion_layer=are_partial_inclusions)

    logger.info("Writing data to {!r}".format(out_file))
    ExclusionsConverter.write_geotiff(out_file, profile, combined_setbacks)


@main.command()
@click.option('--excl_fpath', '-excl', required=True,
              type=click.Path(exists=True),
              help=('Path to .h5 file containing exclusion layers, will also '
                    'be the location of any new setback layers'))
@click.option('--feature_type', '-ft', required=True, type=STR,
              help=('Setback feature type. Must be one of {}'
                    .format(set(SETBACKS.keys()))))
@click.option('--features_path', '-feats', required=True,
              type=click.Path(exists=True),
              help=('Path to:/n'
                    '- State level structure .geojson or directory containing '
                    'geotiff files./n'
                    '- State level roads .gdb or directory containing .gdb '
                    'files./n'
                    '-transmission or railroad CONUS wide .shp file./n'
                    '- State level parcel .gpkg file.'))
@click.option('--out_dir', '-o', required=True, type=STR,
              help=('Directory to save setbacks geotiff(s) into'))
@click.option('--hub_height', '-hh', default=None, type=FLOAT,
              help=('Turbine hub height(m), used along with rotor diameter to '
                    'compute blade tip height which is used to determine '
                    'setback distance. Must be provided if '
                    '`base_setback_dist` is not given'))
@click.option('--rotor_diameter', '-rd', default=None, type=FLOAT,
              help=('Turbine rotor diameter(m), used along with hub height to '
                    'compute blade tip height which is used to determine '
                    'setback distance. Must be provided if '
                    '`base_setback_dist` is not given'))
@click.option('--base_setback_dist', '-bsd', default=None, type=FLOAT,
              help=('Base setback distance value. Must be provided if '
                    'hub_height and rotor_diameter are not given'))
@click.option('--regs_fpath', '-regs', default=None, type=STR,
              show_default=True,
              help=('Path to regulations .csv file, if None create '
                    'generic setbacks using max - tip height * "multiplier", '
                    'by default None'))
@click.option('--multiplier', '-mult', default=None, type=FLOAT,
              show_default=True,
              help=('setback multiplier to use if regulations are not '
                    'supplied, if str, must a key in '
                    '{"high": 3, "moderate": 1.1}, if supplied along with '
                    'regs_fpath, will be ignored, multiplied with max-tip '
                    'height, by default None'))
@click.option('--weights_calculation_upscale_factor', '-wcuf',
              default=None, type=INT, show_default=True,
              help=('Scale factor to use for inclusion weights calculation. '
                    'See the `AbstractBaseSetbacks` documentation for more '
                    'details. By default None.'))
@click.option('--max_workers', '-mw', default=None, type=INT,
              show_default=True,
              help=('Number of workers to use for setback computation, if 1 '
                    'run in serial, if > 1 run in parallel with that many '
                    'workers, if None run in parallel on all available cores, '
                    'by default None'))
@click.option('--replace', '-r', is_flag=True,
              help=('Flag to replace local layer data with arr if layer '
                    'already exists in the exclusion .h5 file'))
@click.option('--hsds', '-hsds', is_flag=True,
              help=('Flag to use h5pyd to handle .h5 domain hosted on AWS '
                    'behind HSDS'))
@click.option('--out_layers', '-ol', type=STR, default=None,
              show_default=True,
              help=('String representation of a dictionary mapping feature '
                    'file names (with extension) to names of layers under '
                    'which exclusions should be saved in the "excl_fpath" '
                    '.h5 file. If "None" or empty dictionary, no layers are '
                    'saved to the h5 file.'))
@click.option('--log_dir', '-log', default=None, type=STR,
              show_default=True,
              help='Directory to dump log files. Default is out_dir.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def local(ctx, excl_fpath, feature_type, features_path, out_dir, hub_height,
          rotor_diameter, base_setback_dist, regs_fpath, multiplier,
          weights_calculation_upscale_factor, max_workers, replace, hsds,
          out_layers, log_dir, verbose):
    """
    Compute Setbacks locally
    """
    if feature_type not in SETBACKS:
        msg = ("feature_type must be one of: {}; got {!r}"
               .format(set(SETBACKS.keys()), feature_type))
        raise RuntimeError(msg)

    if 'VERBOSE' in ctx.obj:
        verbose = any((ctx.obj['VERBOSE'], verbose))

    log_modules = [__name__, 'reVX', 'reV', 'rex']
    init_mult(ctx.obj['NAME'], log_dir, modules=log_modules, verbose=verbose)

    # same check as the config in case someone invokes this from the
    # direct command line instead of a config file for some bizarre reason
    validate_setback_regulations_input(base_setback_dist, hub_height,
                                       rotor_diameter)

    logger.info('Computing setbacks from structures in {}'
                .format(features_path))
    logger.debug('Setbacks to be computed with:\n'
                 '- base_setback_dist = {}\n'
                 '- hub_height = {}\n'
                 '- rotor_diameter = {}\n'
                 '- regs_fpath = {}\n'
                 '- multiplier = {}\n'
                 '- using max_workers = {}\n'
                 '- replace layer if needed = {}\n'
                 '- weights calculation upscale factor = {}\n'
                 '- out_layers = {}\n'
                 .format(base_setback_dist, hub_height, rotor_diameter,
                         regs_fpath, multiplier, max_workers, replace,
                         weights_calculation_upscale_factor, out_layers))

    regulations = select_setback_regulations(base_setback_dist, hub_height,
                                             rotor_diameter, regs_fpath,
                                             multiplier)

    setbacks_class = SETBACKS[feature_type]
    wcuf = weights_calculation_upscale_factor
    if isinstance(out_layers, str):
        out_layers = dict_str_load(out_layers)

    setbacks_class.run(excl_fpath, features_path, out_dir, regulations,
                       weights_calculation_upscale_factor=wcuf,
                       max_workers=max_workers, replace=replace, hsds=hsds,
                       out_layers=out_layers)
    logger.info('Setbacks computed and written to {}'.format(out_dir))


@main.command()
@click.option('--config', '-c', required=True,
              type=click.Path(exists=True),
              help='Filepath to Setbacks config json file.')
@click.pass_context
def from_config(ctx, config):
    """
    Run Setbacks from a config.
    """
    config = SetbacksConfig(config)

    if config.execution_control.option == 'local':
        run_local(ctx, config)

    if config.execution_control.option == 'eagle':
        eagle(config)


def run_local(ctx, config):
    """
    Run Setbacks locally from config

    Parameters
    ----------
    ctx : click.ctx
        click ctx object
    config : `reVX.config.setbacks.SetbacksConfig`
        Setbacks config object.
    """
    ctx.obj['NAME'] = config.name
    wcuf = config.weights_calculation_upscale_factor
    ctx.invoke(local,
               excl_fpath=config.excl_fpath,
               feature_type=config.feature_type,
               features_path=config.features_path,
               out_dir=config.dirout,
               hub_height=config.hub_height,
               rotor_diameter=config.rotor_diameter,
               base_setback_dist=config.base_setback_dist,
               regs_fpath=config.regs_fpath,
               multiplier=config.multiplier,
               weights_calculation_upscale_factor=wcuf,
               max_workers=config.execution_control.max_workers,
               replace=config.replace,
               hsds=config.hsds,
               out_layers=config.out_layers,
               verbose=config.log_level==logging.DEBUG)


def eagle(config):
    """
    Run Setbacks on Eagle HPC.

    Parameters
    ----------
    config : `reVX.config.setbacks.SetbacksConfig`
        Setbacks config object.
    """
    features_path = config.features_path
    cls = SETBACKS[config.feature_type]
    features = cls.get_feature_paths(features_path)
    if not features:
        msg = ('No valid feature files were found at {}!'
               .format(features_path))
        logger.error(msg)
        raise FileNotFoundError(msg)

    for fpath in features:
        fpath_config = deepcopy(config)
        fpath_config['features_path'] = fpath
        launch_job(fpath_config)


def launch_job(config):
    """
    Launch job from config on SLURM

    Parameters
    ----------
    config : `reVX.config.setbacks.SetbacksConfig`
        Setbacks config object.
    """
    log_dir = config.log_directory
    stdout_path = os.path.join(log_dir, 'stdout/')
    name = os.path.basename(config.features_path).split('.')[0]
    name = "{}-{}".format(config.name, name)
    cmd = get_node_cmd(name, config)

    logger.info('Computing Setbacks on Eagle with '
                'node name "{}"'.format(name))
    slurm_manager = SLURM()
    out = slurm_manager.sbatch(cmd,
                               alloc=config.execution_control.allocation,
                               memory=config.execution_control.memory,
                               walltime=config.execution_control.walltime,
                               feature=config.execution_control.feature,
                               name=name, stdout_path=stdout_path,
                               conda_env=config.execution_control.conda_env,
                               module=config.execution_control.module)[0]
    if out:
        msg = ('Kicked off Setbacks job "{}" '
               '(SLURM jobid #{}) on Eagle.'
               .format(name, out))
    else:
        msg = ('Was unable to kick off Setbacks job "{}". '
               'Please see the stdout error messages'
               .format(name))

    click.echo(msg)
    logger.info(msg)


def get_node_cmd(name, config):
    """
    Get the node CLI call for the Setbacks computation

    Parameters
    ----------
    config : `reVX.config.setbacks.SetbacksConfig`
        Setbacks config object.

    Returns
    -------
    cmd : str
        CLI call to submit to SLURM execution.
    """
    wcuf = config.weights_calculation_upscale_factor
    args = ['-n {}'.format(SLURM.s(name)),
            'local',
            '-excl {}'.format(SLURM.s(config.excl_fpath)),
            '-ft {}'.format(SLURM.s(config.feature_type)),
            '-feats {}'.format(SLURM.s(config.features_path)),
            '-o {}'.format(SLURM.s(config.dirout)),
            '-regs {}'.format(SLURM.s(config.regs_fpath)),
            '-mult {}'.format(SLURM.s(config.multiplier)),
            '-wcuf {}'.format(SLURM.s(wcuf)),
            '-ol {}'.format(SLURM.s(config.out_layers)),
            '-mw {}'.format(SLURM.s(config.execution_control.max_workers)),
            '-log {}'.format(SLURM.s(config.log_directory)),
            ]

    if config.base_setback_dist is None:
        args.append('-hh {}'.format(SLURM.s(config.hub_height)))
        args.append('-rd {}'.format(SLURM.s(config.rotor_diameter)))
    else:
        args.append('-bsd {}'.format(SLURM.s(config.base_setback_dist)))

    if config.replace:
        args.append('-r')

    if config.hsds:
        args.append('-hsds')

    if config.log_level == logging.DEBUG:
        args.append('-v')

    cmd = 'python -m reVX.setbacks.setbacks_cli {}'.format(' '.join(args))
    logger.debug('Submitting the following cli call:\n\t{}'.format(cmd))

    return cmd


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running Setbacks CLI')
        raise
