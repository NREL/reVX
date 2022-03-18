# -*- coding: utf-8 -*-
"""
RPM Command Line Interface
"""
import click
import logging
import os
from rex.utilities.loggers import init_mult
from rex.utilities.cli_dtypes import STR, INT
from rex.utilities.hpc import SLURM
from rex.utilities.utilities import dict_str_load, get_class_properties

from reVX.config.rpm import (RPMConfig, ClusterConfigGroup,
                             RepProfilesConfigGroup)
from reVX.rpm.rpm_manager import RPMClusterManager as rpm_cm
from reVX.rpm.rpm_output import RPMOutput as rpm_o
from reVX.utilities.exceptions import RPMRuntimeError
from reVX import __version__

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--name', '-n', default='RPM', type=STR,
              show_default=True,
              help='Job name. Default is "RPM".')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """
    RPM Command Line Interface
    """
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose
    ctx.obj['NAME'] = name


@main.command()
def valid_config_keys():
    """
    Echo the valid RPM config keys
    """
    config_classes = [RPMConfig, ClusterConfigGroup, RepProfilesConfigGroup]
    for cls in config_classes:
        cls_name = str(cls).rsplit('.', maxsplit=1)[-1].strip("'>")
        click.echo("Valid keys for {}: {}"
                   .format(cls_name, ', '.join(get_class_properties(cls))))


def run_local(ctx, config):
    """
    Run reV to ReEDs locally from config

    Parameters
    ----------
    ctx : click.ctx
        click ctx object
    config : RPMConfig
        RPM Config object
    """
    ctx.obj['NAME'] = config.name
    ctx.invoke(local,
               out_dir=config.dirout,
               cf_profiles=config.cf_profiles,
               log_dir=config.log_directory,
               max_workers=config.execution_control.max_workers)

    if config.cluster is not None:
        ctx.invoke(cluster,
                   rpm_meta=config.cluster.rpm_meta,
                   region_col=config.cluster.region_col,
                   dist_rank_filter=config.cluster.dist_rank_filter,
                   contiguous_filter=config.cluster.contiguous_filter)

    if config.rep_profiles is not None:
        ctx.invoke(rep_profiles,
                   rpm_clusters=config.rep_profiles.rpm_clusters,
                   exclusions=config.rep_profiles.exclusions,
                   excl_dict=config.rep_profiles.excl_dict,
                   techmap_dset=config.rep_profiles.techmap_dset,
                   trg_bins=config.rep_profiles.trg_bins,
                   trg_dset=config.rep_profiles.trg_dset,
                   n_profiles=config.rep_profiles.n_profiles,
                   forecast_fpath=config.rep_profiles.forecast_fpath)


@main.command()
@click.option('--config', '-c', required=True,
              type=click.Path(exists=True),
              help='Filepath to reVX-rpm config json file.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config, verbose):
    """
    Run reVX-rpm from a config.
    """

    config = RPMConfig(config)

    if 'VERBOSE' in ctx.obj:
        if any((ctx.obj['VERBOSE'], verbose)):
            config._log_level = logging.DEBUG
    elif verbose:
        config._log_level = logging.DEBUG

    if config.execution_control.option == 'local':
        run_local(ctx, config)
    elif config.execution_control.option == 'eagle':
        eagle(config)


@main.group(chain=True)
@click.option('--out_dir', '-o', required=True, type=click.Path(),
              help='Directory to dump output files')
@click.option('--cf_profiles', '-cf', required=True,
              type=click.Path(exists=True),
              help=('Path to reV .h5 file containing desired capacity factor '
                    'profiles'))
@click.option('--log_dir', '-log', default=None, type=STR, show_default=True,
              help='Directory to dump log files. Default is out_dir.')
@click.option('--max_workers', '-mw', type=INT, default=None,
              show_default=True,
              help=('Number of parallel workers. 1 will run serial, '
                    'None will use all available.'))
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def local(ctx, out_dir, cf_profiles, log_dir, max_workers, verbose):
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

    log_modules = [__name__, 'reVX', 'reV', 'rex']
    init_mult(name, log_dir, modules=log_modules, verbose=verbose)

    logger.info('Running reV to RPM pipeline\n'
                'Outputs to be stored in: {}'.format(out_dir))

    ctx.obj['CF_PROFILES'] = cf_profiles
    ctx.obj['MAX_WORKERS'] = max_workers


@local.command()
@click.option('--rpm_meta', '-m', required=True, type=click.Path(exists=True),
              help='Path to .csv or .json containing the RPM meta data:'
              '- Categorical regions of interest with column label "region"'
              '- # of clusters per region with column label "clusters"'
              '- A column that maps the RPM regions to the cf_fpath meta data:'
              '  "res_gid" (priorized) or "gen_gid". This can be omitted if '
              '  the rpm_region_col kwarg input is found in the cf_fpath meta')
@click.option('--region_col', '-reg', type=str, default=None,
              show_default=True,
              help='The meta-data field to map RPM regions to')
@click.option('--dist_rank_filter', '-drf', is_flag=True,
              help=('Re-cluster data by minimizing the sum of the: '
                    'distance between each point and each cluster centroid'))
@click.option('--contiguous_filter', '-cf', is_flag=True,
              help=('Flag to re-classify clusters by making contigous cluster '
                    'polygons'))
@click.pass_context
def cluster(ctx, rpm_meta, region_col, dist_rank_filter, contiguous_filter):
    """
    Cluster RPM Regions
    """
    name = ctx.obj['NAME']
    cf_profiles = ctx.obj['CF_PROFILES']
    out_dir = ctx.obj['OUT_DIR']
    max_workers = ctx.obj['MAX_WORKERS']

    logger.info('Clustering regions based on:\n{}'.format(rpm_meta))
    rpm_clusters = rpm_cm.run_clusters(cf_profiles, rpm_meta, out_dir,
                                       job_tag=name, rpm_region_col=region_col,
                                       max_workers=max_workers,
                                       dist_rank_filter=dist_rank_filter,
                                       contiguous_filter=contiguous_filter)

    logger.info('reVX - RPM clustering methods complete.')
    ctx.obj['RPM_CLUSTERS'] = rpm_clusters


@local.command()
@click.option('--rpm_clusters', '-rc', type=STR, default=None,
              help=('Path to pre-existing RPM cluster results .csv with '
                    '(gid, gen_gid, cluster_id, rank)'))
@click.option('--exclusions', '-excl', default=None,
              type=click.Path(exists=True), show_default=True,
              help=('Filepath to exclusions data (must match the techmap grid)'
                    ' None will not apply exclusions.'))
@click.option('--excl_dict', '-exd', default=None, type=STR, show_default=True,
              help='String representation of a dictionary of exclusion '
              'LayerMask arguments {layer: {kwarg: value}} where layer is a '
              'dataset in excl_fpath and kwarg can be "inclusion_range", '
              '"exclude_values", "include_values", "use_as_weights", '
              'or "weight".')
@click.option('--techmap_dset', '-tmd', default=None, type=STR,
              show_default=True,
              help=('Dataset name in the techmap file containing the '
                    'exclusions-to-resource mapping data.'))
@click.option('--trg_bins', '-trg', default=None, type=STR, show_default=True,
              help=('Filepath to a single-column CSV containing ordered '
                    'TRG bin edges.'))
@click.option('--trg_dset', '-trgd', default='lcoe_fcr', type=STR,
              show_default=True,
              help=('TRG dataset found in cf_fpath that is associated with '
                    'the TRG bins'))
@click.option('--n_profiles', '-np', type=INT, default=1, show_default=True,
              help=('Number of profiles per cluster to export.'))
@click.option('--forecast_fpath', '-fcst', type=STR, default=None,
              show_default=True,
              help=('reV generation output file for forecast data. If this is '
                    'input, profiles will be taken from forecast file instead '
                    'of the cf file, based on a NN mapping.'))
@click.pass_context
def rep_profiles(ctx, rpm_clusters, exclusions, excl_dict, techmap_dset,
                 trg_bins, trg_dset, n_profiles, forecast_fpath):
    """
    Extract representative profiles from RPM clusters
    """
    name = ctx.obj['NAME']
    cf_profiles = ctx.obj['CF_PROFILES']
    out_dir = ctx.obj['OUT_DIR']
    max_workers = ctx.obj['MAX_WORKERS']
    if rpm_clusters is None:
        if 'RPM_CLUSTERS' not in ctx.obj:
            msg = ('You must run "cluster" or provide path to existing '
                   'RPM clusters to extract representative profiles!')
            logger.error(msg)
            raise RPMRuntimeError(msg)

        rpm_clusters = ctx.obj['RPM_CLUSTERS']

    logger.info('Extracting representative profiles from RPM clusters: {}'
                .format(rpm_clusters))
    logger.info('Extracting representative profiles using exclusions: {}'
                .format(exclusions))

    if trg_bins is not None:
        logger.info('Applying TRGs from dset "{}" : {}'
                    .format(trg_dset, trg_bins))

    if isinstance(excl_dict, str):
        excl_dict = dict_str_load(excl_dict)

    rpm_o.process_outputs(rpm_clusters, cf_profiles, exclusions, excl_dict,
                          techmap_dset, out_dir, job_tag=name,
                          max_workers=max_workers, trg_bins=trg_bins,
                          trg_dset=trg_dset)

    if forecast_fpath is not None or n_profiles > 1:
        logger.info('Extracting extra representative profiles from: {}'
                    .format(rpm_clusters))
        if forecast_fpath is not None:
            logger.info('Using forecast file: {}'.format(forecast_fpath))

        rpm_o.extract_profiles(rpm_clusters, cf_profiles, out_dir,
                               n_profiles=n_profiles, job_tag=name,
                               max_workers=max_workers,
                               forecast_fpath=forecast_fpath)


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
            '-cf {}'.format(SLURM.s(config.cf_profiles)),
            '-mw {}'.format(SLURM.s(config.execution_control.max_workers)),
            '-log {}'.format(SLURM.s(config.log_directory)),
            ]

    if config.log_level == logging.DEBUG:
        args.append('-v')

    if config.cluster is not None:
        cluster = ['cluster',
                   '-m {}'.format(SLURM.s(config.cluster.rpm_meta)),
                   '-reg {}'.format(SLURM.s(config.cluster.region_col)),
                   ]

        if config.cluster.dist_rank_filter:
            cluster.append('-drf')

        if config.cluster.contiguous_filter:
            cluster.append('-cf')

        args.extend(cluster)

    if config.rep_profiles is not None:
        rep_profiles = config.rep_profiles.copy()
        profiles = ['rep_profiles',
                    '-rc {}'.format(SLURM.s(rep_profiles.rpm_clusters)),
                    '-excl {}'.format(SLURM.s(rep_profiles.exclusions)),
                    '-exd {}'.format(SLURM.s(rep_profiles.excl_dict)),
                    '-tmp {}'.format(SLURM.s(rep_profiles.techmap_dset)),
                    '-trg {}'.format(SLURM.s(rep_profiles.trg_bins)),
                    '-trgd {}'.format(SLURM.s(rep_profiles.trg_dset)),
                    '-np {}'.format(SLURM.s(rep_profiles.n_profiles)),
                    '-fcst {}'.format(SLURM.s(rep_profiles.forecast_fpath)),
                    ]

        args.extend(profiles)

    cmd = 'python -m reVX.rpm.rpm_cli {}'.format(' '.join(args))
    logger.debug('Submitting the following cli call:\n\t{}'.format(cmd))

    return cmd


def eagle(config):
    """
    Run reVX-RPM on Eagle HPC.

    Parameters
    ----------
    config : reVX.config.rpm.RPMConfig
        reVX-RPM config object.
    """

    cmd = get_node_cmd(config)
    name = config.name
    log_dir = config.log_directory
    stdout_path = os.path.join(log_dir, 'stdout/')

    logger.info('Running reVX-RPM pipeline on Eagle with '
                'node name "{}"'.format(name))
    slurm_manager = SLURM()
    out = slurm_manager.sbatch(cmd,
                               name=name,
                               stdout_path=stdout_path,
                               alloc=config.execution_control.allocation,
                               memory=config.execution_control.memory,
                               walltime=config.execution_control.walltime,
                               feature=config.execution_control.feature,
                               module=config.execution_control.module,
                               conda_env=config.execution_control.conda_env,
                               )[0]
    if out:
        msg = ('Kicked off reVX-RPM pipeline job "{}" '
               '(SLURM jobid #{}) on Eagle.'
               .format(name, out))
    else:
        msg = ('Was unable to kick off reVX-RPM pipeline job "{}". '
               'Please see the stdout error messages'
               .format(name))

    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running RPM CLI')
        raise
