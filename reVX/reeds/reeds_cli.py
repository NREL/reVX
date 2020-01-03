# -*- coding: utf-8 -*-
"""
ReEDS Command Line Interface
"""
import click
import logging
import os
from reV.utilities.cli_dtypes import STR, STRLIST
from reV.utilities.execution import SLURM, SubprocessManager

from reVX.config.reeds import ReedsConfig
from reVX.reeds.reeds_classification import ReedsClassifier
from reVX.reeds.reeds_profiles import ReedsProfiles
from reVX.reeds.reeds_timeslices import ReedsTimeslices
from reVX.utilities.exceptions import ReedsRuntimeError
from reVX.utilities.loggers import init_mult

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

    if 'verbose' in ctx.obj:
        if any((ctx.obj['verbose'], verbose)):
            config._logging_level = logging.DEBUG
    elif verbose:
        config._logging_level = logging.DEBUG

    if config.execution_control.option == 'local':
        ctx.obj['TABLE'] = config.rev_table
        ctx.obj['OUT_DIR'] = config.dirout

        ctx.invoke(classify, config.classify.resource_classes,
                   config.classify.regions, config.classify.sc_bins,
                   config.classify.cluster_on)

        if config.profiles is not None:
            ctx.invoke(profiles, config.profiles.cf_profiles,
                       config.profiles.n_profiles,
                       config.profiles.profiles_dset,
                       config.profiles.rep_method,
                       config.profiles.err_method,
                       config.profiles.reg_cols,
                       config.profiles.parallel)

        if config.timeslices is not None:
            ctx.invoke(timeslices, config.timeslices.profiles,
                       config.timeslices.timeslices,
                       config.timeslices.reg_cols,
                       config.timeslices.all_profiles)

    if config.execution_control.option == 'eagle':
        eagle(config)


@main.group()
@click.option('--rev_table', '-rt', required=True,
              type=click.Path(exists=True),
              help=('Path to .csv containing reV aggregation or '
                    'supply curve table'))
@click.option('--out_dir', '-o', required=True, type=click.Path(),
              help='Directory to dump output files')
@click.option('--log_dir', '-log', default=None, type=STR,
              help='Directory to dump log files. Default is out_dir.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def local(ctx, rev_table, out_dir, log_dir, verbose):
    """
    Run reVX-REEDS on local hardware.
    """
    ctx.obj['TABLE'] = rev_table
    ctx.obj['OUT_DIR'] = out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if log_dir is None:
        log_dir = out_dir

    name = ctx.obj['NAME']
    if 'verbose' in ctx.obj:
        verbose = any((ctx.obj['verbose'], verbose))

    log_modules = [__name__, 'reVX.reeds', 'reV.rep_profiles']
    init_mult(name, log_dir, modules=log_modules, verbose=verbose)

    logger.info('Running reV to ReEDS pipeline with reV table: {}'
                .format(rev_table))
    logger.info('Outputs to be stored in: {}'.format(out_dir))


@local.group(chain=True, invoke_without_command=True)
@click.option('--resource_classes', '-rc', required=True,
              type=click.Path(exists=True),
              help='.csv or .json containing resource class definitions')
@click.option('--regions', '-r', type=str, default='reeds_region',
              help='Mapping of supply curve points to geographic region')
@click.option('--sc_bins', '-scb', type=int, default=3,
              help=('Number of bins (clusters) to create for each '
                    'region/resource bin combination'))
@click.option('--cluster_on', '-cl', type=str, default='trans_cap_cost',
              help='Column(s) in rev_table to cluster on')
@click.pass_context
def classify(ctx, resource_classes, regions, sc_bins, cluster_on):
    """
    Extract ReEDS (region, bin, class) groups
    """
    name = ctx.obj['NAME']
    rev_table = ctx.obj['TABLE']
    out_dir = ctx.obj['OUT_DIR']

    logger.info('Extracting ReEDS (region, bin, class) groups'
                .format())
    kwargs = {'cluster_on': cluster_on, 'method': 'kmeans'}
    out = ReedsClassifier.create(rev_table, resource_classes,
                                 region_map=regions, sc_bins=sc_bins,
                                 cluster_kwargs=kwargs)
    table_full, table, agg_table = out

    out_path = os.path.join(out_dir, '{}_table.csv'.format(name))
    table.to_csv(out_path, index=False)
    out_path = os.path.join(out_dir, '{}_agg_table.csv'.format(name))
    agg_table.to_csv(out_path, index=False)

    ctx.obj['TABLE'] = table_full

    logger.info('reVX - ReEDS classification methods complete.')


@classify.command()
@click.option('--cf_profiles', '-cf', required=True,
              type=click.Path(exists=True),
              help=('Path to reV .h5 file containing desired capacity factor '
                    'profiles'))
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
@click.option('--reg_cols', '-rcp', type=STRLIST,
              default=('region', 'bin', 'class'),
              help=('Label(s) for a categorical region column(s) to extract '
                    'profiles for (default is region, bin, class)'))
@click.option('--parallel', '-p', is_flag=True,
              help=('Extract profiles in parallel by "group". '
                    'Default is serial.'))
@click.pass_context
def profiles(ctx, cf_profiles, n_profiles, profiles_dset, rep_method,
             err_method, reg_cols, parallel):
    """
    Extract ReEDS represntative profiles
    """
    name = ctx.obj['NAME']
    table = ctx.obj['TABLE']
    out_dir = ctx.obj['OUT_DIR']

    logger.info('Extracting ReEDS representative profiles for {} groups, '
                'from {}.'
                .format(reg_cols, cf_profiles))

    out_path = os.path.join(out_dir, '{}_hourly_cf.h5'.format(name))
    logger.info('Saving representative hourly cf profiles to {}.'
                .format(out_path))

    ReedsProfiles.run(cf_profiles, table, profiles_dset=profiles_dset,
                      rep_method=rep_method, err_method=err_method,
                      n_profiles=n_profiles, reg_cols=reg_cols,
                      parallel=parallel, fout=out_path,
                      hourly=True)

    ctx.obj['PROFILES'] = out_path

    logger.info('reVX - ReEDS representative profile methods complete.')


@classify.command()
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
                                      legacy_format=True)

    out_path = os.path.join(out_dir, '{}_performance.csv'.format(name))
    logger.info('Saving timeslice performance stats to {}'.format(out_path))
    stats.to_csv(out_path, index=False)

    out_path = os.path.join(out_dir, '{}_correlations.csv'
                            .format(name))
    logger.info('Saving timeslice correlations to {}'.format(out_path))
    corr.to_csv(out_path, index=False)

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
    s = SubprocessManager.s

    args = ('-n {name} local -rt {rev_table} -o {out_dir} -log {log_dir} '
            .format(name=s(config.name),
                    rev_table=s(config.rev_table),
                    out_dir=s(config.dirout),
                    log_dir=s(config.logdir)))

    if config.logging_level == logging.DEBUG:
        args += '-v '

    args += ('classify -rc {resource_classes} -r {regions} -scb {sc_bins} '
             '-cl {cluster_on} '
             .format(resource_classes=s(config.classify.resource_classes),
                     regions=s(config.classify.regions),
                     sc_bins=s(config.classify.sc_bins),
                     cluster_on=s(config.classify.cluster_on)))

    if config.profiles is not None:
        args += ('profiles -cf {cf_profiles} -np {n_profiles} '
                 '-pd {profiles_dset} -rm {rep_method} -em {err_method} '
                 '-rcp {reg_cols} '
                 .format(cf_profiles=s(config.profiles.cf_profiles),
                         n_profiles=s(config.profiles.n_profiles),
                         profiles_dset=s(config.profiles.profiles_dset),
                         rep_method=s(config.profiles.rep_method),
                         err_method=s(config.profiles.err_method),
                         reg_cols=s(config.profiles.reg_cols)))
        if config.profiles.parallel:
            args += '-p '

    if config.timeslices is not None:
        args += ('timeslices -pr {profiles} -ts {timeslices} -rct {reg_cols} '
                 .format(profiles=s(config.timeslices.profiles),
                         timeslices=s(config.timeslices.timeslices),
                         reg_cols=s(config.timeslices.reg_cols)))
        if config.timeslices.all_profiles:
            args += '-ap '

    cmd = 'python -m reVX.reeds.reeds_cli {}'.format(args)
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
    slurm = SLURM(cmd, alloc=config.execution_control.alloc,
                  memory=config.execution_control.node_mem,
                  walltime=config.execution_control.walltime,
                  feature=config.execution_control.feature,
                  name=name, stdout_path=stdout_path)
    if slurm.id:
        msg = ('Kicked off reVX-REEDS pipeline job "{}" '
               '(SLURM jobid #{}) on Eagle.'
               .format(name, slurm.id))
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
