# -*- coding: utf-8 -*-
"""
reV-ReEDS-PLEXOS command line interface (cli).
"""
import os
import pandas as pd
import click
import logging

from rex.utilities.hpc import SLURM
from rex.utilities.cli_dtypes import STR, INT, INTLIST, STRLIST
from rex.utilities.loggers import init_mult

from reVX.plexos.rev_reeds_plexos import RevReedsPlexosManager
from reVX import __version__

logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.version_option(version=__version__)
@click.option('--name', '-n', default='plx', type=STR,
              help='Job name. Default is "plx".')
@click.option('--job_input', '-j', required=True,
              type=click.Path(exists=True),
              help=('Path to plexos job input csv.'))
@click.option('--out_dir', '-o', required=True, type=click.Path(),
              help='Directory to dump output files')
@click.option('--cf_years', '-y', required=True, type=INTLIST,
              help='Capacity factor resource year.')
@click.option('--build_years', '-by', required=True, type=INTLIST,
              help='REEDS build years to aggregate profiles for.')
@click.option('--scenario', '-s', default=None, type=STR, show_default=True,
              help='Optional filter to run just one scenario from job input.')
@click.option('--plexos_columns', '-pc', default=None, type=STRLIST,
              show_default=True,
              help='Optional list of additional columns to pass through from '
              'the plexos input table')
@click.option('-ffb', '--force_full_build', is_flag=True,
              help='Flag to ensure the full requested buildout is built at '
              'each SC point. If True, the remainder of the requested build '
              'will always be built at the last resource gid in the sc point.')
@click.option('-fsm', '--force_shape_map', is_flag=True,
              help='Flag to force the mapping of supply curve points to the '
              'plexos node shape file input (if a shape file is input) via '
              'nearest neighbor to shape centroid.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, job_input, out_dir, cf_years, build_years,
         scenario, plexos_columns, force_full_build, force_shape_map, verbose):
    """reV-ReEDS-PLEXOS Command Line Interface"""

    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['JOB_INPUT'] = job_input
    ctx.obj['OUT_DIR'] = out_dir
    ctx.obj['CF_YEARS'] = cf_years
    ctx.obj['BUILD_YEARS'] = build_years
    ctx.obj['SCENARIO'] = scenario
    ctx.obj['PLEXOS_COLUMNS'] = plexos_columns
    ctx.obj['FORCE_FULL_BUILD'] = force_full_build
    ctx.obj['FORCE_SHAPE_MAP'] = force_shape_map
    ctx.obj['VERBOSE'] = verbose

    if ctx.invoked_subcommand is None:
        log_modules = [__name__, 'reVX', 'reV', 'rex']
        init_mult(name, out_dir, modules=log_modules,
                  verbose=verbose)
        logger.info('Running reV to PLEXOS pipeline using job input: {}'
                    .format(job_input))
        logger.info('Outputs to be stored in: {}'.format(out_dir))
        logger.info('Aggregating plexos scenario "{}".'.format(scenario))
        for cf_year in cf_years:
            RevReedsPlexosManager.run(job_input, out_dir, scenario=scenario,
                                      cf_year=cf_year, build_years=build_years,
                                      plexos_columns=plexos_columns,
                                      force_full_build=force_full_build,
                                      force_shape_map=force_shape_map)


def get_node_cmd(name, job_input, out_dir, cf_year, build_year,
                 scenario, plexos_columns, force_full_build,
                 force_shape_map, verbose):
    """Get a CLI call command for the plexos CLI."""

    args = ['-n {}'.format(SLURM.s(name)),
            '-j {}'.format(SLURM.s(job_input)),
            '-o {}'.format(SLURM.s(out_dir)),
            '-y [{}]'.format(SLURM.s(cf_year)),
            '-by [{}]'.format(SLURM.s(build_year)),
            '-s {}'.format(SLURM.s(scenario)),
            '-pc {}'.format(SLURM.s(plexos_columns)),
            ]

    if force_full_build:
        args.append('-ffb')

    if force_shape_map:
        args.append('-fsm')

    if verbose:
        args.append('-v')

    cmd = ('python -m reVX.plexos.rev_reeds_plexos_cli {}'
           .format(' '.join(args)))
    logger.debug('Creating the following command line call:\n\t{}'.format(cmd))

    return cmd


@main.command()
@click.option('--alloc', '-a', required=True, type=STR,
              help='Eagle allocation account name.')
@click.option('--memory', '-mem', default=90, type=INT, show_default=True,
              help='Eagle node memory request in GB. Default is 90')
@click.option('--walltime', '-wt', default=1.0, type=float, show_default=True,
              help='Eagle walltime request in hours. Default is 1.0')
@click.option('--feature', '-l', default=None, type=STR, show_default=True,
              help=('Additional flags for SLURM job. Format is "--qos=high" '
                    'or "--depend=[state:job_id]". Default is None.'))
@click.option('--stdout_path', '-sout', default=None, type=STR,
              show_default=True,
              help='Subprocess standard output path. Default is in out_dir.')
@click.pass_context
def eagle(ctx, alloc, memory, walltime, feature, stdout_path):
    """Eagle submission tool for PLEXOS aggregation."""

    name = ctx.obj['NAME']
    job_input = ctx.obj['JOB_INPUT']
    out_dir = ctx.obj['OUT_DIR']
    cf_years = ctx.obj['CF_YEARS']
    build_years = ctx.obj['BUILD_YEARS']
    scenario = ctx.obj['SCENARIO']
    plexos_columns = ctx.obj['PLEXOS_COLUMNS']
    force_full_build = ctx.obj['FORCE_FULL_BUILD']
    force_shape_map = ctx.obj['FORCE_SHAPE_MAP']
    verbose = ctx.obj['VERBOSE']

    if stdout_path is None:
        stdout_path = os.path.join(out_dir, 'stdout/')

    if scenario is None:
        job = pd.read_csv(job_input)
        scenarios = job.scenario.unique()
    else:
        scenarios = [scenario]

    slurm_manager = ctx.obj.get('SLURM_MANAGER', None)
    if slurm_manager is None:
        slurm_manager = SLURM()
        ctx.obj['SLURM_MANAGER'] = slurm_manager

    for scenario in scenarios:
        for cf_year in cf_years:
            for build_year in build_years:
                node_name = ('{}_{}_{}_{}'
                             .format(name, scenario.replace(' ', '_'),
                                     build_year, cf_year))
                cmd = get_node_cmd(node_name, job_input, out_dir, cf_year,
                                   build_year, scenario, plexos_columns,
                                   force_full_build, force_shape_map, verbose)

                logger.info('Running reVX plexos aggregation on Eagle with '
                            'node name "{}"'.format(node_name))

                out = slurm_manager.sbatch(cmd,
                                           alloc=alloc,
                                           memory=memory,
                                           walltime=walltime,
                                           feature=feature,
                                           name=node_name,
                                           stdout_path=stdout_path)[0]
                if out:
                    msg = ('Kicked off reVX plexos aggregation job "{}" '
                           '(SLURM jobid #{}) on Eagle.'
                           .format(node_name, out))
                else:
                    msg = ('Was unable to kick off reVX plexos aggregation job'
                           ' "{}". Please see the stdout error messages'
                           .format(node_name))

                click.echo(msg)
                logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running PLEXOS CLI')
        raise
