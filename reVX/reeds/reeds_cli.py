# -*- coding: utf-8 -*-
"""
ReEDS Command Line Interface
"""
import click
import logging
import os
from reV.utilities.cli_dtypes import STR, STRLIST
from reV.utilities.execution import SLURM

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


@main.group()
@click.option('--config', '-c', required=True,
              type=click.Path(exists=True),
              help='Filepath to reVX-REEDS config json file.')
def from_config(ctx, config):
    """
    Run reVX-REEDS from a config.
    """
    config = ReedsConfig(config)

    if config.execution_control.option == 'local':
        ctx.obj['TABLE'] = config.rev_table
        ctx.obj['OUT_DIR'] = config.dirout

        classify(config.classify.resource_classes, config.classify.regions,
                 config.classify.n_bins, config.classify.cluster_on)

        if config.profiles is not None:
            profiles(config.profiles.cf_profiles,
                     config.profiles.n_profiles,
                     config.profiles.profiles_dset,
                     config.profiles.rep_method,
                     config.profiles.err_method,
                     config.profiles.reg_cols,
                     config.profiles.parallel)

        if config.timeslices is not None:
            timeslices(config.timeslices.timeslices,
                       config.timeslices.profiles)

    if config.execution_control.option == 'eagle':
        eagle(config)


@main.group()
@click.option('--rev_table', '-rt', required=True,
              type=click.Path(exists=True),
              help=('Path to .csv containing reV aggregation or '
                    'supply curve table'))
@click.option('--out_dir', '-o', required=True, type=click.Path(),
              help='Directory to dump output files')
def local(ctx, rev_table, out_dir):
    """
    Run reVX-REEDS on local hardware.
    """
    ctx.obj['TABLE'] = rev_table
    ctx.obj['OUT_DIR'] = out_dir

    name = ctx.obj['NAME']
    verbose = ctx.obj['verbose']

    init_mult(name, out_dir, modules=[__name__, 'reVX.reeds'],
              verbose=verbose)

    logger.info('Running reV to ReEDS pipeline with reV table: {}'
                .format(rev_table))
    logger.info('Outputs to be stored in: {}'.format(out_dir))


@local.group(chain=True, invoke_without_command=True)
@click.option('--resource_classes', '-rc', required=True,
              type=click.Path(exists=True),
              help='.csv or .json containing resource class definitions')
@click.option('--regions', '-r', type=str, default='reeds_region',
              help='Mapping of supply curve points to geographic region')
@click.option('--n_bins', '-nb', type=int, default=3,
              help=('Number of bins (clusters) to create for each '
                    'region/resource bin combination'))
@click.option('--cluster_on', '-cl', type=str, default='trans_cap_cost',
              help='Column(s) in rev_table to cluster on')
@click.pass_context
def classify(ctx, resource_classes, regions, n_bins, cluster_on):
    """
    Extract ReEDS (region, bin, class) groups
    """
    name = ctx.obj['NAME']
    rev_table = ctx.obj['TABLE']
    out_dir = ctx.obj['OUT_DIR']

    logger.info('Extracting ReEDS (region, bin, class) groups'
                .format())
    kwargs = {'cluster_on': cluster_on, 'method': 'kmeans'}
    table, agg_table = ReedsClassifier.create(rev_table, resource_classes,
                                              region_map=regions,
                                              sc_bins=n_bins,
                                              cluster_kwargs=kwargs)

    out_path = os.path.join(out_dir, '{}_table.csv'.format(name))
    table.to_csv(out_path, index=False)
    out_path = os.path.join(out_dir, '{}_agg_table.csv'.format(name))
    agg_table.to_csv(out_path, index=False)

    ctx.obj['TABLE'] = table


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
@click.option('--reg_cols', '-rcol', type=STRLIST,
              default=('region', 'bin', 'class'),
              help=('Label(s) for a categorical region column(s) to extract '
                    'profiles for'))
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

    out_path = os.path.join(out_dir, '{}_profiles.h5'.format(name))
    logger.info('Saving profiles to {}.'.format(out_path))

    ReedsProfiles.run(cf_profiles, table, profiles_dset=profiles_dset,
                      rep_method=rep_method, err_method=err_method,
                      n_profiles=n_profiles, reg_cols=reg_cols,
                      parallel=parallel, fout=out_path)

    ctx.obj['PROFILES'] = out_path


@classify.command()
@click.option('--timeslices', '-ts', required=True,
              type=click.Path(exists=True),
              help='.csv containing timeslice mapping')
@click.option('--profiles', '-pr', type=click.Path(exists=True),
              default=None,
              help=('Path to .h5 file containing (representative) profiles, '
                    'not needed if chained with profiles command'))
@click.pass_context
def timeslices(ctx, timeslices, profiles):
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
    timeslice_stats = ReedsTimeslices.stats(profiles, timeslices)

    out_path = os.path.join(out_dir, '{}_timeslices-means.csv'.format(name))
    logger.info('Saving timeslice means to {}'.format(out_path))
    timeslice_stats[0].to_csv(out_path)

    out_path = os.path.join(out_dir, '{}_timeslices-stdevs.csv'.format(name))
    logger.info('Saving timeslice stdevs to {}'.format(out_path))
    timeslice_stats[1].to_csv(out_path)


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

    args = ('-n {name} local -rt {rev_table} -o {out_dir} '
            .format(name=config.name,
                    rev_table=config.rev_table,
                    out_dir=config.dirout))

    if config.logging_level.upper() == 'DEBUG':
        args += '-v '

    args += ('classify -rc {resource_classes} -r {regions} -nb {n_bins} '
             '-cl {cluster_on} '
             .format(resource_classes=config.classify.resource_classes,
                     regions=config.classify.regions,
                     n_bins=config.classify.n_bins,
                     cluster_on=config.classify.cluster_on))

    if config.profiles is not None:
        args += ('profiles -cf {cf_profiles} -np {n_profiles} '
                 '-pd {profiles_dset} -rm {rep_method} -em {err_method} '
                 '-rcol {reg_cols} '
                 .format(cf_profiles=config.profiles.cf_profiles,
                         n_profiles=config.profiles.n_profiles,
                         profiles_dset=config.profiles.profiles_dset,
                         rep_method=config.profiles.rep_method,
                         err_method=config.profiles.err_method,
                         reg_cols=config.profiles.reg_cols))
        if config.profiles.parallel:
            args += '-p '

    if config.timeslices is not None:
        args += ('timeslices -ts {timeslices} -pr {profiles} '
                 .format(timeslices=config.timeslices.timeslices,
                         profiles=config.timeslices.profiles))

    cmd = 'python -m reVX.reeds.reeds_cli {}'.format(args)
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
