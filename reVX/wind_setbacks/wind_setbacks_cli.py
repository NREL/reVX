# -*- coding: utf-8 -*-
# pylint: disable=all
"""
Wind Setbacks CLI
"""
import click
from copy import deepcopy
import logging
import os

from rex.utilities.loggers import init_mult
from rex.utilities.cli_dtypes import STR, FLOAT, INT
from rex.utilities.hpc import SLURM
from rex.utilities.utilities import get_class_properties

from reVX.config.wind_setbacks import WindSetbacksConfig
from reVX.wind_setbacks.wind_setbacks import (StructureWindSetbacks,
                                              RoadWindSetbacks,
                                              RailWindSetbacks,
                                              TransmissionWindSetbacks)
from reVX import __version__

logger = logging.getLogger(__name__)


STATE_SETBACKS = {'structure': StructureWindSetbacks,
                  'road': RoadWindSetbacks,
                  'rail': RailWindSetbacks,
                  'tranmission': TransmissionWindSetbacks}


@click.group()
@click.option('--name', '-n', default='WindSetbacks', type=STR,
              show_default=True,
              help='Job name. Default is "WindSetbaks".')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """
    Wind Setbacks Command Line Interface
    """
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['VERBOSE'] = verbose


@main.command()
def version():
    """
    print version
    """
    click.echo(__version__)


@main.command()
def valid_config_keys():
    """
    Echo the valid Wind Setbacks config keys
    """
    click.echo(', '.join(get_class_properties(WindSetbacksConfig)))


@main.command()
@click.option('--config', '-c', required=True,
              type=click.Path(exists=True),
              help='Filepath to Wind Setbacks config json file.')
@click.pass_context
def from_config(ctx, config):
    """
    Run Wind Setbacks from a config.
    """
    config = WindSetbacksConfig(config)

    if config.execution_control.option == 'local':
        run_local(ctx, config)

    if config.execution_control.option == 'eagle':
        eagle(config)


@main.group()
@click.option('--excl_h5', '-excl', required=True,
              type=click.Path(exists=True),
              help=('Path to .h5 file containing exclusion layers, will also '
                    'be the location of any new setback layers'))
@click.option('--features_path', '-feats', required=True,
              type=click.Path(exists=True),
              help=('Path to:/n'
                    '- State level structure .geotiff or directory containing '
                    'geotiff files./n'
                    '- State level roads .gdb or directory containing .gdb '
                    'files./n'
                    '-transmission or railroad CONUS wide .shp file.'))
@click.option('--out_dir', '-o', required=True, type=str,
              help=('Directory to save setbacks geotiff(s) into'))
@click.option('--hub_height', '-height', required=True, type=float,
              help=('Turbine hub height(m), used along with rotor diameter to '
                    'compute blade tip height which is used to determine '
                    'setback distance'))
@click.option('--rotor_diameter', '-diameter', required=True, type=float,
              help=('Turbine rotor diameter(m), used along with hub height to '
                    'compute blade tip height which is used to determine '
                    'setback distance'))
@click.option('--regs_fpath', '-regs', default=None, type=STR,
              show_default=True,
              help=('Path to wind regulations .csv file, if None create '
                    'generic setbacks using max - tip height * "multiplier", '
                    'by default None'))
@click.option('--multiplier', '-mult', default=None, type=FLOAT,
              show_default=True,
              help=('setback multiplier to use if wind regulations are not '
                    'supplied, if str, must a key in '
                    '{"high": 3, "moderate": 1.1}, if supplied along with '
                    'regs_fpath, will be ignored, multiplied with max-tip '
                    'height, by default None'))
@click.option('--max_workers', '-mw', default=None, type=INT,
              show_default=True,
              help=('Number of workers to use for setback computation, if 1 '
                    'run in serial, if > 1 run in parallel with that many '
                    'workers, if None run in parallel on all available cores, '
                    'by default None'))
@click.option('--replace', '-r', is_flag=True,
              help=('Flag to replace local layer data with arr if layer '
                    'already exists in the exlcusion .h5 file'))
@click.option('--hsds', '-hsds', is_flag=True,
              help=('Flag to use h5pyd to handle .h5 domain hosted on AWS '
                    'behind HSDS'))
@click.option('--log_dir', '-log', default=None, type=STR,
              show_default=True,
              help='Directory to dump log files. Default is out_dir.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def local(ctx, excl_h5, features_path, out_dir, hub_height, rotor_diameter,
          regs_fpath, multiplier, max_workers, replace, hsds, log_dir,
          verbose):
    """
    Compute Wind Setbacks locally
    """
    ctx.obj['EXCL_H5'] = excl_h5
    ctx.obj['FEATURES_PATH'] = features_path
    ctx.obj['OUT_DIR'] = out_dir
    ctx.obj['HUB_HEIGHT'] = hub_height
    ctx.obj['ROTOR_DIAMETER'] = rotor_diameter
    ctx.obj['REGS_FPATH'] = regs_fpath
    ctx.obj['MULTIPLIER'] = multiplier
    ctx.obj['MAX_WORKERS'] = max_workers
    ctx.obj['REPLACE'] = replace
    ctx.obj['HSDS'] = hsds

    if 'VERBOSE' in ctx.obj:
        verbose = any((ctx.obj['VERBOSE'], verbose))

    log_modules = [__name__, 'reVX', 'reV', 'rex']
    init_mult(ctx.obj['NAME'], log_dir, modules=log_modules, verbose=verbose)


@local.command()
@click.pass_context
def structure_setbacks(ctx):
    """
    Compute wind setbacks from structures
    """
    excl_h5 = ctx.obj['EXCL_H5']
    features_path = ctx.obj['FEATURES_PATH']
    out_dir = ctx.obj['OUT_DIR']
    hub_height = ctx.obj['HUB_HEIGHT']
    rotor_diameter = ctx.obj['ROTOR_DIAMETER']
    regs_fpath = ctx.obj['REGS_FPATH']
    multiplier = ctx.obj['MULTIPLIER']
    max_workers = ctx.obj['MAX_WORKERS']
    replace = ctx.obj['REPLACE']
    hsds = ctx.obj['HSDS']

    logger.info('Computing setbacks from structures in {}'
                .format(features_path))
    logger.debug('Setbacks to be computed with:\n'
                 '- hub_height = {}\n'
                 '- rotor_diameter = {}\n'
                 '- regs_fpath = {}\n'
                 '- multiplier = {}\n'
                 '- using max_workers = {}\n'
                 '- replace layer if needed = {}\n'
                 .format(hub_height, rotor_diameter, regs_fpath, multiplier,
                         max_workers, replace))

    StructureWindSetbacks.run(excl_h5, features_path, out_dir, hub_height,
                              rotor_diameter, regs_fpath=regs_fpath,
                              multiplier=multiplier, max_workers=max_workers,
                              replace=replace, hsds=hsds)
    logger.info('Setbacks computed and writen to {} as {}'
                .format(excl_h5, out_dir))


@local.command()
@click.pass_context
def road_setbacks(ctx):
    """
    Compute wind setbacks from roads
    """
    excl_h5 = ctx.obj['EXCL_H5']
    features_path = ctx.obj['FEATURES_PATH']
    out_dir = ctx.obj['OUT_DIR']
    hub_height = ctx.obj['HUB_HEIGHT']
    rotor_diameter = ctx.obj['ROTOR_DIAMETER']
    regs_fpath = ctx.obj['REGS_FPATH']
    multiplier = ctx.obj['MULTIPLIER']
    max_workers = ctx.obj['MAX_WORKERS']
    replace = ctx.obj['REPLACE']
    hsds = ctx.obj['HSDS']

    logger.info('Computing setbacks from roads in {}'
                .format(features_path))
    logger.debug('Setbacks to be computed with:\n'
                 '- hub_height = {}\n'
                 '- rotor_diameter = {}\n'
                 '- regs_fpath = {}\n'
                 '- multiplier = {}\n'
                 '- using max_workers = {}\n'
                 '- replace layer if needed = {}\n'
                 .format(hub_height, rotor_diameter, regs_fpath, multiplier,
                         max_workers, replace))

    RoadWindSetbacks.run(excl_h5, features_path, out_dir, hub_height,
                         rotor_diameter, regs_fpath=regs_fpath,
                         multiplier=multiplier, max_workers=max_workers,
                         replace=replace, hsds=hsds)
    logger.info('Setbacks computed and writen to {} as {}'
                .format(excl_h5, out_dir))


@local.command()
@click.pass_context
def transmission_setbacks(ctx):
    """
    Compute wind setbacks from transmission
    """
    excl_h5 = ctx.obj['EXCL_H5']
    features_path = ctx.obj['FEATURES_PATH']
    out_dir = ctx.obj['OUT_DIR']
    hub_height = ctx.obj['HUB_HEIGHT']
    rotor_diameter = ctx.obj['ROTOR_DIAMETER']
    regs_fpath = ctx.obj['REGS_FPATH']
    multiplier = ctx.obj['MULTIPLIER']
    max_workers = ctx.obj['MAX_WORKERS']
    replace = ctx.obj['REPLACE']
    hsds = ctx.obj['HSDS']

    logger.info('Computing setbacks from transmission in {}'
                .format(features_path))
    logger.debug('Setbacks to be computed with:\n'
                 '- hub_height = {}\n'
                 '- rotor_diameter = {}\n'
                 '- regs_fpath = {}\n'
                 '- multiplier = {}\n'
                 '- using max_workers = {}\n'
                 '- replace layer if needed = {}\n'
                 .format(hub_height, rotor_diameter, regs_fpath, multiplier,
                         max_workers, replace))

    TransmissionWindSetbacks.run(excl_h5, features_path, out_dir, hub_height,
                                 rotor_diameter, regs_fpath=regs_fpath,
                                 multiplier=multiplier,
                                 max_workers=max_workers, replace=replace,
                                 hsds=hsds)
    logger.info('Setbacks computed and writen to {} as {}'
                .format(excl_h5, out_dir))


@local.command()
@click.pass_context
def rail_setbacks(ctx):
    """
    Compute wind setbacks from railroads
    """
    excl_h5 = ctx.obj['EXCL_H5']
    features_path = ctx.obj['FEATURES_PATH']
    out_dir = ctx.obj['OUT_DIR']
    hub_height = ctx.obj['HUB_HEIGHT']
    rotor_diameter = ctx.obj['ROTOR_DIAMETER']
    regs_fpath = ctx.obj['REGS_FPATH']
    multiplier = ctx.obj['MULTIPLIER']
    max_workers = ctx.obj['MAX_WORKERS']
    replace = ctx.obj['REPLACE']
    hsds = ctx.obj['HSDS']

    logger.info('Computing setbacks from structures in {}'
                .format(features_path))
    logger.debug('Setbacks to be computed with:\n'
                 '- hub_height = {}\n'
                 '- rotor_diameter = {}\n'
                 '- regs_fpath = {}\n'
                 '- multiplier = {}\n'
                 '- using max_workers = {}\n'
                 '- replace layer if needed = {}\n'
                 .format(hub_height, rotor_diameter, regs_fpath, multiplier,
                         max_workers, replace))

    RailWindSetbacks.run(excl_h5, features_path, out_dir, hub_height,
                         rotor_diameter, regs_fpath=regs_fpath,
                         multiplier=multiplier, max_workers=max_workers,
                         replace=replace, hsds=hsds)
    logger.info('Setbacks computed and writen to {} as {}'
                .format(excl_h5, out_dir))


def run_local(ctx, config):
    """
    Run Wind Setbacks locally from config

    Parameters
    ----------
    ctx : click.ctx
        click ctx object
    config : reVX.config.wind_setbacks.WindSetbacks
        Wind Setbacks config object.
    """
    ctx.obj['NAME'] = config.name
    ctx.invoke(local,
               excl_h5=config.excl_h5,
               features_path=config.features_path,
               out_dir=config.dirout,
               hub_height=config.hub_height,
               rotor_diameter=config.rotor_diameter,
               regs_fpath=config.regs_fpath,
               multiplier=config.multiplier,
               max_workers=config.max_workers,
               replace=config.replace)

    feature_type = config.feature_type
    if feature_type == 'structure':
        ctx.invoke(structure_setbacks)
    elif feature_type == 'road':
        ctx.invoke(road_setbacks)
    elif feature_type == 'transmission':
        ctx.invoke(transmission_setbacks)
    elif feature_type == 'rail':
        ctx.invoke(rail_setbacks)
    else:
        msg = 'Feature type must be one of {}'.format(config.FEATURE_TYPES)
        raise RuntimeError(msg)


def get_node_cmd(name, config):
    """
    Get the node CLI call for the Wind Setbacks computation

    Parameters
    ----------
    config : reVX.config.wind_setbacks.WindSetbacks
        Wind Setbacks config object.

    Returns
    -------
    cmd : str
        CLI call to submit to SLURM execution.
    """
    args = ['-n {}'.format(SLURM.s(name)),
            'local',
            '-excl {}'.format(SLURM.s(config.excl_h5)),
            '-feats {}'.format(SLURM.s(config.features_path)),
            '-o {}'.format(SLURM.s(config.dirout)),
            '-height {}'.format(SLURM.s(config.hub_height)),
            '-diameter {}'.format(SLURM.s(config.rotor_diameter)),
            '-regs {}'.format(SLURM.s(config.regs_fpath)),
            '-mult {}'.format(SLURM.s(config.multiplier)),
            '-mw {}'.format(SLURM.s(config.max_workers)),
            '-log {}'.format(SLURM.s(config.logdir)),
            ]

    if config.replace:
        args.append('-r')

    if config.hsds:
        args.append('-hsds')

    if config.log_level == logging.DEBUG:
        args.append('-v')

    feature_type = config.feature_type
    if feature_type == 'structure':
        args.append('structure-setbacks')
    elif feature_type == 'road':
        args.append('road-setbacks')
    elif feature_type == 'transmission':
        args.append('transmission-setbacks')
    elif feature_type == 'rail':
        args.append('rail-setbacks')
    else:
        msg = 'Feature type must be one of {}'.format(config.FEATURE_TYPES)
        raise RuntimeError(msg)

    cmd = ('python -m reVX.wind_setbacks.wind_setbacks_cli {}'
           .format(' '.join(args)))
    logger.debug('Submitting the following cli call:\n\t{}'.format(cmd))

    return cmd


def launch_job(config):
    """
    Launch job from config on SLURM

    Parameters
    ----------
    config : reVX.config.wind_setbacks.WindSetbacks
        Wind Setbacks config object.
    """
    log_dir = config.logdir
    stdout_path = os.path.join(log_dir, 'stdout/')
    name = os.path.basename(config.features_path).split('.')[0]
    name = "{}-{}".format(config.name, name)
    cmd = get_node_cmd(name, config)

    logger.info('Computing Wind Setbacks on Eagle with '
                'node name "{}"'.format(name))
    slurm_manager = SLURM()
    out = slurm_manager.sbatch(cmd,
                               alloc=config.execution_control.allocation,
                               memory=config.execution_control.memory,
                               walltime=config.execution_control.walltime,
                               feature=config.execution_control.feature,
                               name=name,
                               stdout_path=stdout_path,
                               conda_env=config.execution_control.conda_env,
                               module=config.execution_control.module)[0]
    if out:
        msg = ('Kicked off Wind Setbacks job "{}" '
               '(SLURM jobid #{}) on Eagle.'
               .format(name, out))
    else:
        msg = ('Was unable to kick off Wind Setbacks job "{}". '
               'Please see the stdout error messages'
               .format(name))

    click.echo(msg)
    logger.info(msg)


def eagle(config):
    """
    Run Wind Setbacks on Eagle HPC.

    Parameters
    ----------
    config : reVX.config.wind_setbacks.WindSetbacks
        Wind Setbacks config object.
    """
    features_path = config.features_path
    cls = STATE_SETBACKS[config.feature_type]
    features = cls._get_feature_paths(features_path)
    if not features:
        msg = ('No valid feature files were found at {}!'
               .format(features_path))
        logger.error(msg)
        raise FileNotFoundError(msg)

    for fpath in features:
        fpath_config = deepcopy(config)
        fpath_config['features_path'] = fpath
        launch_job(fpath_config)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running Wind Setbacks CLI')
        raise
