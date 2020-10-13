# -*- coding: utf-8 -*-
"""
PLEXOS Plants command line interface (cli).
"""
import click
import logging
import os

from rex.utilities.cli_dtypes import STR, INT
from rex.utilities.hpc import SLURM
from rex.utilities.loggers import init_mult

from reVX.plexos.plexos_plants import PlantProfileAggregation

logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.option('--name', '-n', default='plexos-plants', type=STR,
              help='Job name. Default is "plexos-plants".')
@click.option('--plexos_table', '-pt', required=True,
              type=click.Path(exists=True),
              help=('Path to PLEXOS table of bus locations and capacity .csv'))
@click.option('--sc_table', '-st', required=True,
              type=click.Path(exists=True),
              help=('Path to Supply Curve table .csv'))
@click.option('--cf_fpath', '-cf', required=True,
              type=click.Path(exists=True),
              help=('Path to reV Generation output .h5 file'))
@click.option('--out_dir', '-out', required=True, type=click.Path(),
              help='Directory to dump output files')
@click.option('--dist_percentile', '-dp', type=int, default=90,
              help=('Percentile to use to compute distance threshold using '
                    'sc_gid to SubStation distance, by default 90'))
@click.option('--lcoe_col', '-lc', type=str, default='total_lcoe',
              help="LCOE column to sort by, by default 'total_lcoe'")
@click.option('--lcoe_thresh', '-lt', type=float, default=1.3,
              help=('LCOE threshold multiplier, exclude sc_gids above '
                    'threshold, by default 1.3'))
@click.option('--max_workers', '-mw', type=INT, default=None,
              help=('Number of workers to use for point and plant creation, '
                    '1 == serial, > 1 == parallel, None == parallel using all '
                    'available cpus, by default None'))
@click.option('--points_per_worker', type=int, default=400,
              help='Number of points to create on each worker, by default 400')
@click.option('--plants_per_worker', type=int, default=40,
              help=('Number of plants to identify on each worker, by default '
                    '40'))
@click.option('-o', '--offshore', is_flag=True,
              help='Include offshore points, by default False')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, plexos_table, sc_table, cf_fpath, out_dir, dist_percentile,
         lcoe_col, lcoe_thresh, max_workers, points_per_worker,
         plants_per_worker, offshore, verbose):
    """PLEXOS plant Command Line Interface"""

    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['PLEXOS_TABLE'] = plexos_table
    ctx.obj['SC_TABLE'] = sc_table
    ctx.obj['CF_FPATH'] = cf_fpath
    ctx.obj['OUT_DIR'] = out_dir
    ctx.obj['DIST_PERCENTILE'] = dist_percentile
    ctx.obj['LCOE_COL'] = lcoe_col
    ctx.obj['LCOE_THRESH'] = lcoe_thresh
    ctx.obj['MAX_WORKERS'] = max_workers
    ctx.obj['POINTS_PER_WORKER'] = points_per_worker
    ctx.obj['PLANTS_PER_WORKER'] = plants_per_worker
    ctx.obj['OFFSHORE'] = offshore
    ctx.obj['VERBOSE'] = verbose

    if ctx.invoked_subcommand is None:
        init_mult(name, out_dir, modules=[__name__,
                                          'reVX.plexos.plexos_plants',
                                          'reVX.handlers.sc_points'],
                  verbose=verbose)
        logger.info('Aggregating Plant for buses in PLEXOS table: {}'
                    .format(plexos_table))
        out_fpath = '-' + os.path.basename(cf_fpath).split('_')[-1]
        out_fpath = os.path.basename(plexos_table).replace('.csv', out_fpath)
        out_fpath = os.path.join(out_dir, out_fpath)
        logger.info('Saving Aggregated Plant Profiles to {}'
                    .format(out_fpath))
        PlantProfileAggregation.run(plexos_table, sc_table, cf_fpath,
                                    out_fpath, dist_percentile=dist_percentile,
                                    lcoe_col=lcoe_col, lcoe_thresh=lcoe_thresh,
                                    max_workers=max_workers,
                                    points_per_worker=points_per_worker,
                                    plants_per_worker=plants_per_worker,
                                    offshore=offshore)


def get_node_cmd(name, plexos_table, sc_table, cf_fpath, out_dir,
                 dist_percentile, lcoe_col, lcoe_thresh, max_workers,
                 points_per_worker, plants_per_worker, offshore, verbose):
    """Build PLEXOS Plant CLI command."""

    args = ['-n {}'.format(SLURM.s(name)),
            '-pt {}'.format(SLURM.s(plexos_table)),
            '-sc {}'.format(SLURM.s(sc_table)),
            '-cf {}'.format(SLURM.s(cf_fpath)),
            '-out {}'.format(SLURM.s(out_dir)),
            '-dp {}'.format(SLURM.s(dist_percentile)),
            '-lc {}'.format(SLURM.s(lcoe_col)),
            '-lt {}'.format(SLURM.s(lcoe_thresh)),
            '-mw {}'.format(SLURM.s(max_workers)),
            '--points_per_worker={}'.format(SLURM.s(points_per_worker)),
            '--plants_per_worker={}'.format(SLURM.s(plants_per_worker)),
            ]

    if offshore:
        args.append('-o')

    if verbose:
        args.append('-v')

    cmd = 'python -m reVX.plexos.plexos_plants_cli {}'.format(' '.join(args))
    logger.debug('Creating the following command line call:\n\t{}'.format(cmd))

    return cmd


@main.command()
@click.option('--alloc', '-a', required=True, type=STR,
              help='Eagle allocation account name.')
@click.option('--walltime', '-wt', default=1.0, type=float,
              help='Eagle walltime request in hours. Default is 1.0')
@click.option('--feature', '-l', default=None, type=STR,
              help=('Additional flags for SLURM job. Format is "--qos=high" '
                    'or "--depend=[state:job_id]". Default is None.'))
@click.option('--stdout_path', '-sout', default=None, type=STR,
              help='Subprocess standard output path. Default is in out_dir.')
@click.pass_context
def eagle(ctx, alloc, walltime, feature, stdout_path):
    """Eagle submission tool for PLEXOS aggregation."""

    name = ctx.obj['NAME']
    plexos_table = ctx.obj['PLEXOS_TABLE']
    sc_table = ctx.obj['SC_TABLE']
    cf_fpath = ctx.obj['CF_FPATH']
    out_dir = ctx.obj['OUT_DIR']
    dist_percentile = ctx.obj['DIST_PERCENTILE']
    lcoe_col = ctx.obj['LCOE_COL']
    lcoe_thresh = ctx.obj['LCOE_THRESH']
    max_workers = ctx.obj['MAX_WORKERS']
    points_per_worker = ctx.obj['POINTS_PER_WORKER']
    plants_per_worker = ctx.obj['PLANTS_PER_WORKER']
    offshore = ctx.obj['OFFSHORE']
    verbose = ctx.obj['VERBOSE']

    if stdout_path is None:
        stdout_path = os.path.join(out_dir, 'stdout/')

    slurm_manager = ctx.obj.get('SLURM_MANAGER', None)
    if slurm_manager is None:
        slurm_manager = SLURM()
        ctx.obj['SLURM_MANAGER'] = slurm_manager

    cmd = get_node_cmd(name, plexos_table, sc_table, cf_fpath, out_dir,
                       dist_percentile, lcoe_col, lcoe_thresh, max_workers,
                       points_per_worker, plants_per_worker, offshore, verbose)

    logger.info('Running reVX plexos plant aggregation on Eagle with '
                'node name "{}"'.format(name))

    out = slurm_manager.sbatch(cmd, alloc=alloc, walltime=walltime,
                               feature=feature, name=name,
                               stdout_path=stdout_path)[0]
    if out:
        msg = ('Kicked off reVX plexos aggregation job "{}" '
               '(SLURM jobid #{}) on Eagle.'
               .format(name, out))
    else:
        msg = ('Was unable to kick off reVX plexos aggregation job "{}". '
               'Please see the stdout error messages'
               .format(name))

    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running PLEXOS-Plant CLI')
        raise
