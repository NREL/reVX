# -*- coding: utf-8 -*-
"""
RPM Command Line Interface
"""
import json
import click
import logging
from reV.utilities.cli_dtypes import STR
from reV.utilities.loggers import init_mult

from reVX.rpm.rpm_manager import RPMClusterManager as rpm_cm
from reVX.rpm.rpm_output import RPMOutput as rpm_o

logger = logging.getLogger(__name__)


@click.group()
@click.option('--name', '-n', default='RPM', type=STR,
              help='Job name. Default is "RPM".')
@click.option('--cf_profiles', '-cf', required=True,
              type=click.Path(exists=True),
              help=('Path to reV .h5 file containing desired capacity factor '
                    'profiles'))
@click.option('--out_dir', '-o', required=True, type=click.Path(),
              help='Directory to dump output files')
@click.option('--parallel', '-p', is_flag=True,
              help='Run clustering in parallel by region. Default is serial.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, cf_profiles, out_dir, parallel, verbose):
    """
    RPM Command Line Interface
    """
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['CF_FPATH'] = cf_profiles
    ctx.obj['OUT_DIR'] = out_dir
    ctx.obj['PARALLEL'] = parallel

    init_mult(name, out_dir, modules=[__name__, 'reVX.rpm'],
              verbose=verbose)

    logger.info('Running reV to RPM pipeline using CF profiles: {}'
                .format(cf_profiles))
    logger.info('Outputs to be stored in: {}'.format(out_dir))


@main.group(invoke_without_command=True)
@click.option('--rpm_meta', '-m', required=True, type=click.Path(exists=True),
              help='.csv or .json containing the RPM meta data')
@click.option('--region_col', '-rc', type=str, default=None,
              help='The meta-data field to map RPM regions to')
@click.pass_context
def cluster(ctx, rpm_meta, region_col):
    """
    Cluster RPM Regions
    """
    name = ctx.obj['NAME']
    cf_fpath = ctx.obj['CF_FPATH']
    out_dir = ctx.obj['OUT_DIR']
    parallel = ctx.obj['PARALLEL']
    ctx.obj['RPM_META'] = rpm_meta
    ctx.obj['META_COL'] = region_col

    if ctx.invoked_subcommand is None:
        logger.info('Clustering regions based on:\n{}'
                    .format(rpm_meta))
        rpm_cm.run_clusters(cf_fpath, rpm_meta, out_dir, job_tag=name,
                            rpm_region_col=region_col, parallel=parallel)


@cluster.command()
@click.option('--exclusions', '-e', default=None,
              type=click.Path(exists=True),
              help=('Filepath to exclusions data (must match the techmap grid)'
                    ' None will not apply exclusions.'))
@click.option('--excl_dict', '-exd', default=None, type=STR,
              help='String representation of a dictionary of exclusion '
              'LayerMask arguments {layer: {kwarg: value}} where layer is a '
              'dataset in excl_fpath and kwarg can be "inclusion_range", '
              '"exclude_values", "include_values", "use_as_weights", '
              'or "weight".')
@click.option('--techmap_dset', '-tmd', default=None, type=STR,
              help=('Dataset name in the exclusions file containing the '
                    'exclusions-to-resource mapping data.'))
@click.option('--trg', '-trg', default=None, type=STR,
              help=('Filepath to TRG LCOE bins.'))
@click.pass_context
def and_profiles(ctx, exclusions, excl_dict, techmap_dset, trg):
    """
    Cluster RPM Regions and extract representative profiles
    """
    name = ctx.obj['NAME']
    cf_fpath = ctx.obj['CF_FPATH']
    out_dir = ctx.obj['OUT_DIR']
    parallel = ctx.obj['PARALLEL']
    rpm_meta = ctx.obj['RPM_META']
    region_col = ctx.obj['META_COL']
    out_dir = ctx.obj['OUT_DIR']
    parallel = ctx.obj['PARALLEL']

    logger.info('Clustering regions based on: {}'.format(rpm_meta))
    logger.info('Extracting representative profiles using exclusions: {}'
                .format(exclusions))

    if isinstance(excl_dict, str):
        excl_dict = excl_dict.replace('\'', '\"')
        excl_dict = excl_dict.replace('None', 'null')
        excl_dict = excl_dict.replace('True', 'true')
        excl_dict = excl_dict.replace('False', 'false')
        excl_dict = json.loads(excl_dict)

    output_kwargs = {}
    if trg is not None:
        output_kwargs = {'trg': trg}
        logger.info('Applying TRGs: {}'.format(trg))

    rpm_cm.run_clusters_and_profiles(cf_fpath, rpm_meta, exclusions,
                                     excl_dict, techmap_dset, out_dir,
                                     job_tag=name, rpm_region_col=region_col,
                                     parallel=parallel,
                                     output_kwargs=output_kwargs)


@main.group(invoke_without_command=True)
@click.option('--rpm_clusters', '-rc', required=True,
              type=click.Path(exists=True),
              help=('Pre-existing RPM cluster results .csv with '
                    '(gid, gen_gid, cluster_id, rank)'))
@click.option('--exclusions', '-e', default=None,
              type=click.Path(exists=True),
              help=('Filepath to exclusions data (must match the techmap grid)'
                    ' None will not apply exclusions.'))
@click.option('--excl_dict', '-exd', default=None, type=STR,
              help='String representation of a dictionary of exclusion '
              'LayerMask arguments {layer: {kwarg: value}} where layer is a '
              'dataset in excl_fpath and kwarg can be "inclusion_range", '
              '"exclude_values", "include_values", "use_as_weights", '
              'or "weight".')
@click.option('--techmap_dset', '-tmd', default=None, type=STR,
              help=('Dataset name in the techmap file containing the '
                    'exclusions-to-resource mapping data.'))
@click.option('--trg', '-trg', default=None, type=STR,
              help=('Filepath to TRG LCOE bins.'))
@click.pass_context
def rep_profiles(ctx, rpm_clusters, exclusions, excl_dict, techmap_dset, trg):
    """
    Extract representative profiles from existing RPM clusters
    """
    name = ctx.obj['NAME']
    cf_fpath = ctx.obj['CF_FPATH']
    out_dir = ctx.obj['OUT_DIR']
    parallel = ctx.obj['PARALLEL']
    ctx.obj['RPM_CLUSTERS'] = rpm_clusters

    if ctx.invoked_subcommand is None:
        logger.info('Extracting representative profiles from RPM clusters: {}'
                    .format(rpm_clusters))
        logger.info('Extracting representative profiles using exclusions: {}'
                    .format(exclusions))

        if trg is not None:
            logger.info('Applying TRGs: {}'.format(trg))

        if isinstance(excl_dict, str):
            excl_dict = excl_dict.replace('\'', '\"')
            excl_dict = excl_dict.replace('None', 'null')
            excl_dict = json.loads(excl_dict)

        rpm_o.process_outputs(rpm_clusters, cf_fpath, exclusions,
                              excl_dict, techmap_dset, out_dir,
                              job_tag=name, parallel=parallel, trg=trg)


@rep_profiles.command()
@click.option('--profiles', '-np', required=True, type=int,
              help=('Number of profiles per cluster to export.'))
@click.option('--forecast_fpath', '-fo', type=STR, default=None,
              help=('reV generation output file for forecast data. If this is '
                    'input, profiles will be taken from forecast file instead '
                    'of the cf file, based on a NN mapping.'))
@click.pass_context
def extra_profiles(ctx, profiles, forecast_fpath):
    """
    Cluster RPM Regions and extract representative profiles
    """
    name = ctx.obj['NAME']
    cf_fpath = ctx.obj['CF_FPATH']
    out_dir = ctx.obj['OUT_DIR']
    parallel = ctx.obj['PARALLEL']
    rpm_clusters = ctx.obj['RPM_CLUSTERS']

    logger.info('Extracting extra representative profiles from: {}'
                .format(rpm_clusters))
    if forecast_fpath is not None:
        logger.info('Using forecast file: {}'.format(forecast_fpath))

    rpm_o.extract_profiles(rpm_clusters, cf_fpath, out_dir,
                           n_profiles=profiles, job_tag=name,
                           parallel=parallel, forecast_fpath=forecast_fpath)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running RPM CLI')
        raise
