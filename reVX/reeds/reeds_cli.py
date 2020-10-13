# -*- coding: utf-8 -*-
"""
ReEDS Command Line Interface
"""
import click
import logging
import os

from rex.utilities.loggers import init_mult
from rex.utilities.cli_dtypes import STR, STRLIST, INT
from rex.utilities.hpc import SLURM
from rex.utilities.utilities import dict_str_load, get_class_properties

from reVX.config.reeds import (ReedsConfig, ClassifyConfigGroup,
                               ProfilesConfigGroup, TimeslicesConfigGroup)
from reVX.reeds.reeds_classification import ReedsClassifier
from reVX.reeds.reeds_profiles import ReedsProfiles
from reVX.reeds.reeds_timeslices import ReedsTimeslices
from reVX.utilities.exceptions import ReedsRuntimeError


logger = logging.getLogger(__name__)


@click.group()
@click.option('--name', '-n', default='ReEDS', type=STR,
              help='Job name. Default is "ReEDS".')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """
    ReEDS Command Line Interface
    """
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['VERBOSE'] = verbose


@main.command()
def valid_config_keys():
    """
    Echo the valid ReEDS config keys
    """
    config_classes = [ReedsConfig, ClassifyConfigGroup,
                      ProfilesConfigGroup, TimeslicesConfigGroup]
    for cls in config_classes:
        cls_name = str(cls).split('.')[-1].strip("'>")
        click.echo("Valid keys for {}: {}"
                   .format(cls_name, ', '.join(get_class_properties(cls))))


def run_local(ctx, config):
    """
    Run reV to ReEDs locally from config

    Parameters
    ----------
    ctx : click.ctx
        click ctx object
    config : ReedsConfig
        Reeds Config object
    """
    ctx.obj['NAME'] = config.name
    ctx.invoke(local,
               out_dir=config.dirout,
               log_dir=config.logdir)

    if config.classify is not None:
        ctx.invoke(classify,
                   rev_table=config.classify.rev_table,
                   resource_classes=config.classify.resource_classes,
                   regions=config.classify.regions,
                   cap_bins=config.classify.cap_bins,
                   sort_bins_by=config.classify.sort_bins_by,
                   pre_filter=config.classify.pre_filter)

    if config.profiles is not None:
        ctx.invoke(profiles,
                   reeds_table=config.profiles.reeds_table,
                   cf_profiles=config.profiles.cf_profiles,
                   gid_col=config.profiles.gid_col,
                   n_profiles=config.profiles.n_profiles,
                   profiles_dset=config.profiles.profiles_dset,
                   rep_method=config.profiles.rep_method,
                   err_method=config.profiles.err_method,
                   weight=config.profiles.weight,
                   reg_cols=config.profiles.reg_cols,
                   max_workers=config.profiles.max_workers)

    if config.timeslices is not None:
        ctx.invoke(timeslices,
                   profiles=config.timeslices.profiles,
                   timeslices=config.timeslices.timeslices,
                   reg_cols=config.timeslices.reg_cols,
                   all_profiles=config.timeslices.all_profiles)


@main.command()
@click.option('--config', '-c', required=True,
              type=click.Path(exists=True),
              help='Filepath to reVX-REEDS config json file.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config, verbose):
    """
    Run reVX-REEDS from a config.
    """

    config = ReedsConfig(config)

    if 'VERBOSE' in ctx.obj:
        if any((ctx.obj['VERBOSE'], verbose)):
            config._log_level = logging.DEBUG
    elif verbose:
        config._log_level = logging.DEBUG

    if config.execution_control.option == 'local':
        run_local(ctx, config)

    if config.execution_control.option == 'eagle':
        eagle(config)


@main.group(chain=True)
@click.option('--out_dir', '-o', required=True, type=click.Path(),
              help='Directory to dump output files')
@click.option('--log_dir', '-log', default=None, type=STR,
              help='Directory to dump log files. Default is out_dir.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def local(ctx, out_dir, log_dir, verbose):
    """
    Run reVX-REEDS on local hardware.
    """
    ctx.obj['OUT_DIR'] = out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if log_dir is None:
        log_dir = out_dir

    name = ctx.obj['NAME']
    if 'VERBOSE' in ctx.obj:
        verbose = any((ctx.obj['VERBOSE'], verbose))

    log_modules = [__name__, 'reVX.reeds', 'reV.rep_profiles']
    init_mult(name, log_dir, modules=log_modules, verbose=verbose)

    logger.info('Running reV to ReEDS pipeline\n'
                'Outputs to be stored in: {}'.format(out_dir))


@local.command()
@click.option('--rev_table', '-rt', required=True,
              type=click.Path(exists=True),
              help=('Path to .csv containing reV aggregation or '
                    'supply curve table'))
@click.option('--resource_classes', '-rc', required=True,
              type=click.Path(exists=True),
              help=("resource_classes: str | pandas.DataFrame\n"
                    "Resource classes, either provided in a .csv, .json or a "
                    "DataFrame\n"
                    "Allowable columns:\n"
                    "- 'class' -> class labels to use\n"
                    "- 'TRG_cap' -> TRG capacity bins to use to create TRG "
                    "classes\n"
                    "- any column in 'rev_table' -> Categorical bins\n"
                    "- '*_min' and '*_max' where * is a numberical column in "
                    "'rev_table' -> Range binning\n"
                    "NOTE: 'TRG_cap' can only be combined with categorical "
                    "bins"))
@click.option('--regions', '-r', type=str, default='reeds_region',
              help='Mapping of supply curve points to geographic region')
@click.option('--cap_bins', '-cb', type=int, default=5,
              help=('Number of capacity bins to create for each '
                    'region/resource bin combination'))
@click.option('--sort_bins_by', '-sb', type=str, default='trans_cap_cost',
              help='Column(s) in rev_table to sort before binning')
@click.option('--pre_filter', '-f', type=STR, default=None,
              help='Column value pair(s) to filter on. If None do not filter')
@click.pass_context
def classify(ctx, rev_table, resource_classes, regions, cap_bins, sort_bins_by,
             pre_filter):
    """
    Extract ReEDS (region, bin, class) groups
    """
    name = ctx.obj['NAME']
    out_dir = ctx.obj['OUT_DIR']

    logger.info('Extracting ReEDS (region, bin, class) groups using '
                'reV sc table {}'.format(rev_table))
    if isinstance(pre_filter, str):
        pre_filter = dict_str_load(pre_filter)

    out = ReedsClassifier.create(rev_table, resource_classes,
                                 region_map=regions, cap_bins=cap_bins,
                                 sort_bins_by=sort_bins_by,
                                 pre_filter=pre_filter)
    table_full, table, agg_table_full, agg_table = out

    out_path = os.path.join(out_dir,
                            '{}_supply_curve_raw_full.csv'.format(name))
    table_full.to_csv(out_path, index=False)
    out_path = os.path.join(out_dir, '{}_supply_curve_raw.csv'.format(name))
    table.to_csv(out_path, index=False)
    out_path = os.path.join(out_dir, '{}_supply_curve_full.csv'.format(name))
    agg_table_full.to_csv(out_path, index=False)
    out_path = os.path.join(out_dir, '{}_supply_curve.csv'.format(name))
    agg_table.to_csv(out_path, index=False)

    ctx.obj['TABLE'] = table_full

    logger.info('reVX - ReEDS classification methods complete.')


@local.command()
@click.option('--reeds_table', '-rt', type=STR, default=None,
              help=('Path to .csv containing reeds classification table '
                    'not needed if chained with classify command'))
@click.option('--cf_profiles', '-cf', required=True,
              type=click.Path(exists=True),
              help=('Path to reV .h5 file containing desired capacity factor '
                    'profiles'))
@click.option('--gid_col', '-gc', type=str, default='gen_gids',
              help='Column label in rev_summary that contains the generation '
              'gids (data index in cf_profiles file path).')
@click.option('--n_profiles', '-np', type=int, default=1,
              help='Number of profiles to extract per "group".')
@click.option('--profiles_dset', '-pd', type=str, default="cf_profile",
              help='Profiles dataset name in cf_profiles file.')
@click.option('--rep_method', '-rm', type=STR, default='meanoid',
              help=('Method identifier for calculation of the representative '
                    'profile.'))
@click.option('--err_method', '-em', type=STR, default='rmse',
              help=('Method identifier for calculation of error from the '
                    'representative profile.'))
@click.option('--weight', '-w', type=str, default='gid_counts',
              help='Column in rev_summary used to apply weighted mean to '
              'profiles. The supply curve table data in the weight column '
              'should have weight values corresponding to the gid_col in '
              'the same row.')
@click.option('--reg_cols', '-rcp', type=STRLIST,
              default=('region', 'bin', 'class'),
              help=('Label(s) for a categorical region column(s) to extract '
                    'profiles for (default is region, bin, class)'))
@click.option('--max_workers', '-mw', type=INT, default=None,
              help=('Number of parallel workers. 1 will run serial, '
                    'None will use all available.'))
@click.pass_context
def profiles(ctx, reeds_table, cf_profiles, gid_col, n_profiles, profiles_dset,
             rep_method, err_method, weight, reg_cols, max_workers):
    """
    Extract ReEDS represntative profiles
    """
    name = ctx.obj['NAME']
    if reeds_table is None:
        reeds_table = ctx.obj['TABLE']

    out_dir = ctx.obj['OUT_DIR']

    logger.info('Extracting ReEDS representative profiles for {} groups, '
                'from {}.'
                .format(reg_cols, cf_profiles))

    out_path = os.path.join(out_dir, '{}_hourly_cf.h5'.format(name))
    logger.info('Saving representative hourly cf profiles to {}.'
                .format(out_path))

    ReedsProfiles.run(cf_profiles, reeds_table,
                      gid_col=gid_col,
                      profiles_dset=profiles_dset,
                      rep_method=rep_method,
                      err_method=err_method,
                      n_profiles=n_profiles,
                      weight=weight,
                      reg_cols=reg_cols,
                      max_workers=max_workers,
                      fout=out_path,
                      hourly=True)

    ctx.obj['PROFILES'] = out_path

    logger.info('reVX - ReEDS representative profile methods complete.')


@local.command()
@click.option('--profiles', '-pr', type=STR, default=None,
              help=('Path to .h5 file containing (representative) profiles, '
                    'not needed if chained with profiles command'))
@click.option('--timeslices', '-ts', required=True,
              type=click.Path(exists=True),
              help='.csv containing timeslice mapping')
@click.option('--reg_cols', '-rct', type=STRLIST,
              default=('region', 'class'),
              help=('Label(s) for a categorical region column(s) to create '
                    'timeslice stats for (default is region and class)'))
@click.option('--all_profiles', '-ap', is_flag=True,
              help='Flag to calculate timeslice stats from all CF profiles '
              'as opposed to representative profiles (default).')
@click.pass_context
def timeslices(ctx, profiles, timeslices, reg_cols, all_profiles):
    """
    Extract timeslices from representative profiles
    """
    name = ctx.obj['NAME']
    out_dir = ctx.obj['OUT_DIR']

    if profiles is None:
        if 'PROFILES' not in ctx.obj:
            msg = ('You must run "profiles" or provide path to existing '
                   'profiles to extract timeslices!')
            logger.error(msg)
            raise ReedsRuntimeError(msg)

        profiles = ctx.obj['PROFILES']

    logger.info('Extracting timeslices from {} using mapping {}'
                .format(profiles, timeslices))

    if all_profiles:
        rev_table = ctx.obj['rev_table']
        logger.info('Extracting timeslice stats from all cf_profiles '
                    'in rev_table (and not just representative profiles).')
    else:
        rev_table = None
        logger.info('Extracting timeslice stats from representative profiles '
                    '(and not all cf_profiles).')

    stats, corr = ReedsTimeslices.run(profiles, timeslices,
                                      rev_table=rev_table,
                                      reg_cols=reg_cols,
                                      legacy_format=False)

    out_path = os.path.join(out_dir, '{}_performance.csv'.format(name))
    logger.info('Saving timeslice performance stats to {}'.format(out_path))
    stats.to_csv(out_path, index=False)

    out_path = os.path.join(out_dir, '{}_correlations.h5'.format(name))
    logger.info('Saving timeslice correlations to {}'.format(out_path))
    ReedsTimeslices.save_correlation_dict(corr, reg_cols, out_path)

    logger.info('reVX - ReEDS timeslice methods complete.')


def get_node_cmd(config):
    """
    Get the node CLI call for the reVX-REEDS pipeline.

    Parameters
    ----------
    config : reVX.config.reeds.ReedsConfig
        reVX-REEDS config object.

    Returns
    -------
    cmd : str
        CLI call to submit to SLURM execution.
    """

    args = ['-n {}'.format(SLURM.s(config.name)),
            'local',
            '-o {}'.format(SLURM.s(config.dirout)),
            '-log {}'.format(SLURM.s(config.logdir)),
            ]

    if config.log_level == logging.DEBUG:
        args.append('-v')

    if config.classify is not None:
        classify = ['classify',
                    '-rt {}'.format(SLURM.s(config.classify.rev_table)),
                    '-rc {}'.format(SLURM.s(config.classify.resource_classes)),
                    '-r {}'.format(SLURM.s(config.classify.regions)),
                    '-cb {}'.format(SLURM.s(config.classify.cap_bins)),
                    '-sb {}'.format(SLURM.s(config.classify.sort_bins_by)),
                    '-f {}'.format(SLURM.s(config.classify.pre_filter)),
                    ]

        args.extend(classify)

    if config.profiles is not None:
        profiles = ['profiles',
                    '-rt {}'.format(SLURM.s(config.profiles.reeds_table)),
                    '-cf {}'.format(SLURM.s(config.profiles.cf_profiles)),
                    '-gc {}'.format(SLURM.s(config.profiles.gid_col)),
                    '-np {}'.format(SLURM.s(config.profiles.n_profiles)),
                    '-pd {}'.format(SLURM.s(config.profiles.profiles_dset)),
                    '-rm {}'.format(SLURM.s(config.profiles.rep_method)),
                    '-em {}'.format(SLURM.s(config.profiles.err_method)),
                    '-w {}'.format(SLURM.s(config.profiles.weight)),
                    '-rcp {}'.format(SLURM.s(config.profiles.reg_cols)),
                    '-mw {}'.format(SLURM.s(config.profiles.max_workers)),
                    ]

        args.extend(profiles)

    if config.timeslices is not None:
        timeslices = ['timeslices',
                      '-pr {}'.format(SLURM.s(config.timeslices.profiles)),
                      '-ts {}'.format(SLURM.s(config.timeslices.timeslices)),
                      '-rct {}'.format(SLURM.s(config.timeslices.reg_cols)),
                      ]

        if config.timeslices.all_profiles:
            timeslices.append('-ap')

        args.extend(timeslices)

    cmd = 'python -m reVX.reeds.reeds_cli {}'.format(' '.join(args))
    logger.debug('Submitting the following cli call:\n\t{}'.format(cmd))

    return cmd


def eagle(config):
    """
    Run reVX-REEDS on Eagle HPC.

    Parameters
    ----------
    config : reVX.config.reeds.ReedsConfig
        reVX-REEDS config object.
    """

    cmd = get_node_cmd(config)
    name = config.name
    log_dir = config.logdir
    stdout_path = os.path.join(log_dir, 'stdout/')

    logger.info('Running reVX-REEDS pipeline on Eagle with '
                'node name "{}"'.format(name))
    slurm_manager = SLURM()
    out = slurm_manager.sbatch(cmd,
                               alloc=config.execution_control.alloc,
                               memory=config.execution_control.node_mem,
                               walltime=config.execution_control.walltime,
                               feature=config.execution_control.feature,
                               name=name, stdout_path=stdout_path,
                               conda_env=config.execution_control.conda_env,
                               module=config.execution_control.module)[0]
    if out:
        msg = ('Kicked off reVX-REEDS pipeline job "{}" '
               '(SLURM jobid #{}) on Eagle.'
               .format(name, out))
    else:
        msg = ('Was unable to kick off reVX-REEDS pipeline job "{}". '
               'Please see the stdout error messages'
               .format(name))

    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running ReEDS CLI')
        raise
