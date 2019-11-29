# -*- coding: utf-8 -*-
"""
ReEDS Command Line Interface
"""
import click
import logging
import os
from reV.utilities.cli_dtypes import STR, STRLIST

from reVX.reeds.reeds_classification import ReedsClassifier
from reVX.reeds.reeds_profiles import ReedsProfiles
from reVX.reeds.reeds_timeslices import ReedsTimeslices
from reVX.utilities.exceptions import ReedsRuntimeError
from reVX.utilities.loggers import init_mult

logger = logging.getLogger(__name__)


@click.group()
@click.option('--name', '-n', default='ReEDS', type=STR,
              help='Job name. Default is "ReEDS".')
@click.option('--rev_table', '-rt', required=True,
              type=click.Path(exists=True),
              help=('Path to .csv containing reV aggregation or '
                    'supply curve table'))
@click.option('--out_dir', '-o', required=True, type=click.Path(),
              help='Directory to dump output files')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, rev_table, out_dir, verbose):
    """
    ReEDS Command Line Interface
    """
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['TABLE'] = rev_table
    ctx.obj['OUT_DIR'] = out_dir

    init_mult(name, out_dir, modules=[__name__, 'reVX.reeds'],
              verbose=verbose)

    logger.info('Running reV to ReEDS pipeline with reV table: {}'
                .format(rev_table))
    logger.info('Outputs to be stored in: {}'.format(out_dir))


@main.group(chain=True, invoke_without_command=True)
@click.option('--bins', '-b', required=True, type=click.Path(exists=True),
              help='.csv or .json containing resource bins')
@click.option('--regions', '-r', type=str, default='reeds_region',
              help='Mapping of supply curve points to geographic region')
@click.option('--classes', '-c', type=int, default=3,
              help=('Number of classes (clusters) to create for each '
                    'region-bin'))
@click.option('--cluster_on', '-col', type=str, default=None,
              help='Columns in rev_table to cluster on')
@click.pass_context
def classify(ctx, bins, regions, classes, cluster_on):
    """
    Extract ReEDS (region, bin, class) groups
    """
    name = ctx.obj['NAME']
    table = ctx.obj['TABLE']
    out_dir = ctx.obj['OUT_DIR']

    logger.info('Extracting ReEDS (region, bin, class) groups'
                .format())
    kwargs = {'cluster_on': cluster_on, 'method': 'kmeans'}
    table = ReedsClassifier.create(table, bins, region_map=regions,
                                   classes=classes, cluster_kwargs=kwargs)

    out_path = os.path.join(out_dir, '{}_table.csv'.format(name))
    table.to_csv(out_path, index=False)

    ctx.obj['TABLE'] = table


@classify.command()
@click.option('--cf_profiles', '-cf', required=True,
              type=click.Path(exists=True),
              help=('Path to reV .h5 file containing desired capacity factor '
                    'profiles'))
@click.option('--n_profiles', '-np', type=int, default=1,
              help='Number of profiles to extract per "group".')
@click.option('--rep_method', '-rm', type=STR, default='meanoid',
              help=('Method identifier for calculation of the representative '
                    'profile.'))
@click.option('--err_method', '-em', type=STR, default='rmse',
              help=('Method identifier for calculation of error from the '
                    'representative profile.'))
@click.option('--reg_cols', '-rcol', type=STRLIST,
              default="('region', 'bin', 'class')",
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
@click.option('--rep_profiles', '-rp', type=click.Path(exists=True),
              default=None,
              help=('Path to .h5 file containing representative profiles, '
                    'not needed if chained with profiles command'))
@click.pass_context
def timeslices(ctx, timeslices, rep_profiles):
    """
    Extract timeslices from representative profiles
    """
    name = ctx.obj['NAME']
    out_dir = ctx.obj['OUT_DIR']
    if rep_profiles is None:
        if 'PROFILES' not in ctx.obj:
            msg = ('You must run "profiles" or provide path to existing '
                   'profiles to extract timeslices!')
            logger.error(msg)
            raise ReedsRuntimeError(msg)

        rep_profiles = ctx.obj['PROFILES']

    logger.info('Extractin timeslices from {} using mapping {}'
                .format(rep_profiles, timeslices))
    timeslice_stats = ReedsTimeslices.stats(rep_profiles, timeslices)

    out_path = os.path.join(out_dir, '{}_timeslices-means.csv'.format(name))
    logger.info('Saving timeslice means to {}'.format(out_path))
    timeslice_stats[0].to_csv(out_path)

    out_path = os.path.join(out_dir, '{}_timeslices-stdevs.csv'.format(name))
    logger.info('Saving timeslice stdevs to {}'.format(out_path))
    timeslice_stats[1].to_csv(out_path)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running ReEDS CLI')
        raise
