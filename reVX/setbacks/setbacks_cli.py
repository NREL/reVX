# -*- coding: utf-8 -*-
# pylint: disable=all
"""
Setbacks CLI
"""
import click
from copy import deepcopy
import logging
import os

from rex.utilities.loggers import init_mult
from rex.utilities.cli_dtypes import STR, FLOAT, INT
from rex.utilities.hpc import SLURM
from rex.utilities.utilities import get_class_properties

from reVX.config.setbacks import SetbacksConfig
from reVX.setbacks import (StructureWindSetbacks,
                           RoadWindSetbacks,
                           RailWindSetbacks,
                           TransmissionWindSetbacks,
                           SolarParcelSetbacks,
                           WindParcelSetbacks,
                           SolarWaterSetbacks,
                           WindWaterSetbacks)
from reVX import __version__

logger = logging.getLogger(__name__)


STATE_SETBACKS = {'structure': StructureWindSetbacks,
                  'road': RoadWindSetbacks,
                  'rail': RailWindSetbacks,
                  'transmission': TransmissionWindSetbacks,
                  'parcel': SolarParcelSetbacks,
                  'water': SolarWaterSetbacks}


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


@main.group()
@click.option('--excl_fpath', '-excl', required=True,
              type=click.Path(exists=True),
              help=('Path to .h5 file containing exclusion layers, will also '
                    'be the location of any new setback layers'))
@click.option('--features_path', '-feats', required=True,
              type=click.Path(exists=True),
              help=('Path to:/n'
                    '- State level structure .geojson or directory containing '
                    'geotiff files./n'
                    '- State level roads .gdb or directory containing .gdb '
                    'files./n'
                    '-transmission or railroad CONUS wide .shp file./n'
                    '- State level parcel .gpkg file.'))
@click.option('--out_dir', '-o', required=True, type=str,
              help=('Directory to save setbacks geotiff(s) into'))
@click.option('--hub_height', '-hh', default=None, type=float,
              help=('Turbine hub height(m), used along with rotor diameter to '
                    'compute blade tip height which is used to determine '
                    'setback distance. Must be provided if '
                    '`base_setback_dist` is not given'))
@click.option('--rotor_diameter', '-rd', default=None, type=float,
              help=('Turbine rotor diameter(m), used along with hub height to '
                    'compute blade tip height which is used to determine '
                    'setback distance. Must be provided if '
                    '`base_setback_dist` is not given'))
@click.option('--base_setback_dist', '-bsd', default=None, type=float,
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
                    'See the `BaseSetbacks` documentation for more details. '
                    'By default None.'))
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
@click.option('--log_dir', '-log', default=None, type=STR,
              show_default=True,
              help='Directory to dump log files. Default is out_dir.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def local(ctx, excl_fpath, features_path, out_dir, hub_height, rotor_diameter,
          base_setback_dist, regs_fpath, multiplier,
          weights_calculation_upscale_factor, max_workers, replace, hsds,
          log_dir, verbose):
    """
    Compute Setbacks locally
    """
    ctx.obj['excl_fpath'] = excl_fpath
    ctx.obj['FEATURES_PATH'] = features_path
    ctx.obj['OUT_DIR'] = out_dir
    ctx.obj['HUB_HEIGHT'] = hub_height
    ctx.obj['ROTOR_DIAMETER'] = rotor_diameter
    ctx.obj['BASE_SETBACK_DIST'] = base_setback_dist
    ctx.obj['REGS_FPATH'] = regs_fpath
    ctx.obj['MULTIPLIER'] = multiplier
    ctx.obj['MAX_WORKERS'] = max_workers
    ctx.obj['REPLACE'] = replace
    ctx.obj['HSDS'] = hsds
    ctx.obj['UPSCALE_FACTOR'] = weights_calculation_upscale_factor

    if 'VERBOSE' in ctx.obj:
        verbose = any((ctx.obj['VERBOSE'], verbose))

    log_modules = [__name__, 'reVX', 'reV', 'rex']
    init_mult(ctx.obj['NAME'], log_dir, modules=log_modules, verbose=verbose)

    # same check as the config in case someone invokes this from the
    # direct command line instead of a config file for some bizarre reason
    no_base_setback = base_setback_dist is None
    invalid_turbine_specs = rotor_diameter is None or hub_height is None

    not_enough_info = no_base_setback and invalid_turbine_specs
    too_much_info = not no_base_setback and not invalid_turbine_specs
    if not_enough_info or too_much_info:
        raise RuntimeError(
            "Must provide either `base_setback_dist` or both `rotor_diameter` "
            "and `hub_height` (but not all three)."
        )


@local.command()
@click.pass_context
def structure_setbacks(ctx):
    """
    Compute setbacks from structures
    """
    excl_fpath = ctx.obj['excl_fpath']
    features_path = ctx.obj['FEATURES_PATH']
    out_dir = ctx.obj['OUT_DIR']
    hub_height = ctx.obj['HUB_HEIGHT']
    rotor_diameter = ctx.obj['ROTOR_DIAMETER']
    regs_fpath = ctx.obj['REGS_FPATH']
    multiplier = ctx.obj['MULTIPLIER']
    max_workers = ctx.obj['MAX_WORKERS']
    replace = ctx.obj['REPLACE']
    hsds = ctx.obj['HSDS']
    wcuf = ctx.obj['UPSCALE_FACTOR']

    logger.info('Computing setbacks from structures in {}'
                .format(features_path))
    logger.debug('Setbacks to be computed with:\n'
                 '- hub_height = {}\n'
                 '- rotor_diameter = {}\n'
                 '- regs_fpath = {}\n'
                 '- multiplier = {}\n'
                 '- using max_workers = {}\n'
                 '- replace layer if needed = {}\n'
                 '- weights calculation upscale factor = {}\n'
                 .format(hub_height, rotor_diameter, regs_fpath, multiplier,
                         max_workers, replace, wcuf))

    StructureWindSetbacks.run(excl_fpath, features_path, out_dir, hub_height,
                              rotor_diameter, regulations_fpath=regs_fpath,
                              multiplier=multiplier,
                              weights_calculation_upscale_factor=wcuf,
                              max_workers=max_workers,
                              replace=replace, hsds=hsds)
    logger.info('Setbacks computed and written to {}'.format(out_dir))


@local.command()
@click.pass_context
def road_setbacks(ctx):
    """
    Compute setbacks from roads
    """
    excl_fpath = ctx.obj['excl_fpath']
    features_path = ctx.obj['FEATURES_PATH']
    out_dir = ctx.obj['OUT_DIR']
    hub_height = ctx.obj['HUB_HEIGHT']
    rotor_diameter = ctx.obj['ROTOR_DIAMETER']
    regs_fpath = ctx.obj['REGS_FPATH']
    multiplier = ctx.obj['MULTIPLIER']
    max_workers = ctx.obj['MAX_WORKERS']
    replace = ctx.obj['REPLACE']
    hsds = ctx.obj['HSDS']
    wcuf = ctx.obj['UPSCALE_FACTOR']

    logger.info('Computing setbacks from roads in {}'
                .format(features_path))
    logger.debug('Setbacks to be computed with:\n'
                 '- hub_height = {}\n'
                 '- rotor_diameter = {}\n'
                 '- regs_fpath = {}\n'
                 '- multiplier = {}\n'
                 '- using max_workers = {}\n'
                 '- replace layer if needed = {}\n'
                 '- weights calculation upscale factor = {}\n'
                 .format(hub_height, rotor_diameter, regs_fpath, multiplier,
                         max_workers, replace, wcuf))

    RoadWindSetbacks.run(excl_fpath, features_path, out_dir, hub_height,
                         rotor_diameter, regulations_fpath=regs_fpath,
                         multiplier=multiplier,
                         weights_calculation_upscale_factor=wcuf,
                         max_workers=max_workers,
                         replace=replace, hsds=hsds)
    logger.info('Setbacks computed and written to {}'.format(out_dir))


@local.command()
@click.pass_context
def transmission_setbacks(ctx):
    """
    Compute setbacks from transmission
    """
    excl_fpath = ctx.obj['excl_fpath']
    features_path = ctx.obj['FEATURES_PATH']
    out_dir = ctx.obj['OUT_DIR']
    hub_height = ctx.obj['HUB_HEIGHT']
    rotor_diameter = ctx.obj['ROTOR_DIAMETER']
    regs_fpath = ctx.obj['REGS_FPATH']
    multiplier = ctx.obj['MULTIPLIER']
    max_workers = ctx.obj['MAX_WORKERS']
    replace = ctx.obj['REPLACE']
    hsds = ctx.obj['HSDS']
    wcuf = ctx.obj['UPSCALE_FACTOR']

    logger.info('Computing setbacks from transmission in {}'
                .format(features_path))
    logger.debug('Setbacks to be computed with:\n'
                 '- hub_height = {}\n'
                 '- rotor_diameter = {}\n'
                 '- regs_fpath = {}\n'
                 '- multiplier = {}\n'
                 '- using max_workers = {}\n'
                 '- replace layer if needed = {}\n'
                 '- weights calculation upscale factor = {}\n'
                 .format(hub_height, rotor_diameter, regs_fpath, multiplier,
                         max_workers, replace, wcuf))

    TransmissionWindSetbacks.run(excl_fpath, features_path, out_dir,
                                 hub_height, rotor_diameter,
                                 regulations_fpath=regs_fpath,
                                 multiplier=multiplier,
                                 weights_calculation_upscale_factor=wcuf,
                                 max_workers=max_workers,
                                 replace=replace,
                                 hsds=hsds)
    logger.info('Setbacks computed and written to {}'.format(out_dir))


@local.command()
@click.pass_context
def rail_setbacks(ctx):
    """
    Compute setbacks from railroads
    """
    excl_fpath = ctx.obj['excl_fpath']
    features_path = ctx.obj['FEATURES_PATH']
    out_dir = ctx.obj['OUT_DIR']
    hub_height = ctx.obj['HUB_HEIGHT']
    rotor_diameter = ctx.obj['ROTOR_DIAMETER']
    regs_fpath = ctx.obj['REGS_FPATH']
    multiplier = ctx.obj['MULTIPLIER']
    max_workers = ctx.obj['MAX_WORKERS']
    replace = ctx.obj['REPLACE']
    hsds = ctx.obj['HSDS']
    wcuf = ctx.obj['UPSCALE_FACTOR']

    logger.info('Computing setbacks from structures in {}'
                .format(features_path))
    logger.debug('Setbacks to be computed with:\n'
                 '- hub_height = {}\n'
                 '- rotor_diameter = {}\n'
                 '- regs_fpath = {}\n'
                 '- multiplier = {}\n'
                 '- using max_workers = {}\n'
                 '- replace layer if needed = {}\n'
                 '- weights calculation scale factor = {}\n'
                 .format(hub_height, rotor_diameter, regs_fpath, multiplier,
                         max_workers, replace, wcuf))

    RailWindSetbacks.run(excl_fpath, features_path, out_dir, hub_height,
                         rotor_diameter, regulations_fpath=regs_fpath,
                         multiplier=multiplier,
                         weights_calculation_upscale_factor=wcuf,
                         max_workers=max_workers,
                         replace=replace, hsds=hsds)
    logger.info('Setbacks computed and written to {}'.format(out_dir))


@local.command()
@click.pass_context
def parcel_setbacks(ctx):
    """
    Compute setbacks from parcels
    """
    excl_fpath = ctx.obj['excl_fpath']
    features_path = ctx.obj['FEATURES_PATH']
    out_dir = ctx.obj['OUT_DIR']
    base_setback_dist = ctx.obj['BASE_SETBACK_DIST']
    regs_fpath = ctx.obj['REGS_FPATH']
    multiplier = ctx.obj['MULTIPLIER']
    max_workers = ctx.obj['MAX_WORKERS']
    replace = ctx.obj['REPLACE']
    hsds = ctx.obj['HSDS']
    wcuf = ctx.obj['UPSCALE_FACTOR']

    logger.info('Computing setbacks from parcels in {}'.format(features_path))

    if base_setback_dist is None:
        # hub-height and rotor diameter guaranteed to exist if
        # `base_setback_dist = None`` due to check performed in `local`
        hub_height = ctx.obj['HUB_HEIGHT']
        rotor_diameter = ctx.obj['ROTOR_DIAMETER']
        logger.debug('Setbacks to be computed with:\n'
                     '- hub_height = {}\n'
                     '- rotor_diameter = {}\n'
                     '- regs_fpath = {}\n'
                     '- multiplier = {}\n'
                     '- using max_workers = {}\n'
                     '- replace layer if needed = {}\n'
                     '- weights calculation upscale factor = {}\n'
                     .format(hub_height, rotor_diameter, regs_fpath,
                             multiplier, max_workers, replace, wcuf))

        WindParcelSetbacks.run(excl_fpath, features_path, out_dir,
                               hub_height=hub_height,
                               rotor_diameter=rotor_diameter,
                               regulations_fpath=regs_fpath,
                               multiplier=multiplier,
                               weights_calculation_upscale_factor=wcuf,
                               max_workers=max_workers, replace=replace,
                               hsds=hsds)
    else:
        logger.debug('Setbacks to be computed with:\n'
                     '- base_setback_dist = {}\n'
                     '- regs_fpath = {}\n'
                     '- multiplier = {}\n'
                     '- using max_workers = {}\n'
                     '- replace layer if needed = {}\n'
                     '- weights calculation upscale factor = {}\n'
                     .format(base_setback_dist, regs_fpath, multiplier,
                             max_workers, replace, wcuf))

        SolarParcelSetbacks.run(excl_fpath, features_path, out_dir,
                                base_setback_dist,
                                regulations_fpath=regs_fpath,
                                multiplier=multiplier,
                                weights_calculation_upscale_factor=wcuf,
                                max_workers=max_workers, replace=replace,
                                hsds=hsds)
    logger.info('Setbacks computed and written to {}'.format(out_dir))


@local.command()
@click.pass_context
def water_setbacks(ctx):
    """
    Compute setbacks from water
    """
    excl_fpath = ctx.obj['excl_fpath']
    features_path = ctx.obj['FEATURES_PATH']
    out_dir = ctx.obj['OUT_DIR']
    base_setback_dist = ctx.obj['BASE_SETBACK_DIST']
    regs_fpath = ctx.obj['REGS_FPATH']
    multiplier = ctx.obj['MULTIPLIER']
    max_workers = ctx.obj['MAX_WORKERS']
    replace = ctx.obj['REPLACE']
    hsds = ctx.obj['HSDS']
    wcuf = ctx.obj['UPSCALE_FACTOR']

    logger.info('Computing setbacks from water in {}'
                .format(features_path))

    if base_setback_dist is None:
        # hub-height and rotor diameter guaranteed to exist if
        # `base_setback_dist = None`` due to check performed in `local`
        hub_height = ctx.obj['HUB_HEIGHT']
        rotor_diameter = ctx.obj['ROTOR_DIAMETER']
        logger.debug('Setbacks to be computed with:\n'
                     '- hub_height = {}\n'
                     '- rotor_diameter = {}\n'
                     '- regs_fpath = {}\n'
                     '- multiplier = {}\n'
                     '- using max_workers = {}\n'
                     '- replace layer if needed = {}\n'
                     '- weights calculation upscale factor = {}\n'
                     .format(hub_height, rotor_diameter, regs_fpath,
                             multiplier, max_workers, replace, wcuf))

        WindWaterSetbacks.run(excl_fpath, features_path, out_dir,
                              hub_height=hub_height,
                              rotor_diameter=rotor_diameter,
                              regulations_fpath=regs_fpath,
                              multiplier=multiplier,
                              weights_calculation_upscale_factor=wcuf,
                              max_workers=max_workers, replace=replace,
                              hsds=hsds)
    else:
        logger.debug('Setbacks to be computed with:\n'
                     '- base_setback_dist = {}\n'
                     '- regs_fpath = {}\n'
                     '- multiplier = {}\n'
                     '- using max_workers = {}\n'
                     '- replace layer if needed = {}\n'
                     '- weights calculation upscale factor = {}\n'
                     .format(base_setback_dist, regs_fpath, multiplier,
                             max_workers, replace, wcuf))

        SolarWaterSetbacks.run(excl_fpath, features_path, out_dir,
                               base_setback_dist,
                               regulations_fpath=regs_fpath,
                               multiplier=multiplier,
                               weights_calculation_upscale_factor=wcuf,
                               max_workers=max_workers, replace=replace,
                               hsds=hsds)

    logger.info('Setbacks computed and written to {}'.format(out_dir))


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
               features_path=config.features_path,
               out_dir=config.dirout,
               hub_height=config.hub_height,
               rotor_diameter=config.rotor_diameter,
               base_setback_dist=config.base_setback_dist,
               regs_fpath=config.regs_fpath,
               multiplier=config.multiplier,
               weights_calculation_upscale_factor=wcuf,
               max_workers=config.execution_control.max_workers,
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
    elif feature_type == 'parcel':
        ctx.invoke(parcel_setbacks)
    elif feature_type == 'water':
        ctx.invoke(water_setbacks)
    else:
        options = set(config.FEATURE_TYPE_EXTRA_REQUIREMENTS.keys())
        msg = 'Feature type must be one of {}; got {}'.format(
            options, feature_type
        )
        raise RuntimeError(msg)


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
            '-feats {}'.format(SLURM.s(config.features_path)),
            '-o {}'.format(SLURM.s(config.dirout)),
            '-regs {}'.format(SLURM.s(config.regs_fpath)),
            '-mult {}'.format(SLURM.s(config.multiplier)),
            '-wcuf {}'.format(SLURM.s(wcuf)),
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

    feature_type = config.feature_type
    if feature_type in STATE_SETBACKS:
        args.append('{}-setbacks'.format(feature_type))
    else:
        options = set(STATE_SETBACKS.keys())
        msg = 'Feature type must be one of {}; got {}'.format(
            options, feature_type
        )
        raise RuntimeError(msg)

    cmd = ('python -m reVX.setbacks.setbacks_cli {}'
           .format(' '.join(args)))
    logger.debug('Submitting the following cli call:\n\t{}'.format(cmd))

    return cmd


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
                               name=name,
                               stdout_path=stdout_path,
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


def eagle(config):
    """
    Run Setbacks on Eagle HPC.

    Parameters
    ----------
    config : `reVX.config.setbacks.SetbacksConfig`
        Setbacks config object.
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
        logger.exception('Error running Setbacks CLI')
        raise
