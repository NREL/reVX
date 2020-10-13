# -*- coding: utf-8 -*-
"""
reV-ReEDS-PLEXOS command line interface (cli).
"""
import os
import pandas as pd
import click
import logging

from rex.utilities.hpc import SLURM
from rex.utilities.cli_dtypes import STR, INT, INTLIST
from rex.utilities.loggers import init_mult

from reVX.plexos.rev_reeds_plexos import Manager

logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.option('--name', '-n', default='plx', type=STR,
              help='Job name. Default is "plx".')
@click.option('--job_input', '-j', required=True,
              type=click.Path(exists=True),
              help=('Path to plexos job input csv.'))
@click.option('--out_dir', '-o', required=True, type=click.Path(),
              help='Directory to dump output files')
@click.option('--reeds_dir', '-rd', required=True, type=click.Path(),
              help='Directory containing REEDS buildout files.')
@click.option('--cf_years', '-y', required=True, type=INTLIST,
              help='Capacity factor resource year.')
@click.option('--build_years', '-by', required=True, type=INTLIST,
              help='REEDS build years to aggregate profiles for.')
@click.option('--scenario', '-s', required=False, default=None, type=STR,
              help='Optional filter to run just one scenario from job input.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, job_input, out_dir, reeds_dir, cf_years, build_years,
         scenario, verbose):
    """reV-ReEDS-PLEXOS Command Line Interface"""

    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['JOB_INPUT'] = job_input
    ctx.obj['OUT_DIR'] = out_dir
    ctx.obj['REEDS_DIR'] = reeds_dir
    ctx.obj['CF_YEARS'] = cf_years
    ctx.obj['BUILD_YEARS'] = build_years
    ctx.obj['SCENARIO'] = scenario
    ctx.obj['VERBOSE'] = verbose

    if ctx.invoked_subcommand is None:
        init_mult(name, out_dir, modules=[__name__, 'reVX.plexos'],
                  verbose=verbose)
        logger.info('Running reV to PLEXOS pipeline using job input: {}'
                    .format(job_input))
        logger.info('Outputs to be stored in: {}'.format(out_dir))
        logger.info('Aggregating plexos scenario "{}".'.format(scenario))
        for cf_year in cf_years:
            Manager.run(job_input, out_dir, reeds_dir, scenario=scenario,
                        cf_year=cf_year, build_years=build_years)


def get_node_cmd(name, job_input, out_dir, reeds_dir, cf_year, build_year,
                 scenario, verbose):
    """Get a CLI call command for the plexos CLI."""

    args = ['-n {}'.format(SLURM.s(name)),
            '-j {}'.format(SLURM.s(job_input)),
            '-o {}'.format(SLURM.s(out_dir)),
            '-rd {}'.format(SLURM.s(reeds_dir)),
            '-y [{}]'.format(SLURM.s(cf_year)),
            '-by [{}]'.format(SLURM.s(build_year)),
            '-s {}'.format(SLURM.s(scenario)),
            ]

    if verbose:
        args.append('-v')

    cmd = 'python -m reVX.plexos.plexos_cli {}'.format(' '.join(args))
    logger.debug('Creating the following command line call:\n\t{}'.format(cmd))

    return cmd


@main.command()
@click.option('--alloc', '-a', required=True, type=STR,
              help='Eagle allocation account name.')
@click.option('--memory', '-mem', default=90, type=INT,
              help='Eagle node memory request in GB. Default is 90')
@click.option('--walltime', '-wt', default=1.0, type=float,
              help='Eagle walltime request in hours. Default is 1.0')
@click.option('--feature', '-l', default=None, type=STR,
              help=('Additional flags for SLURM job. Format is "--qos=high" '
                    'or "--depend=[state:job_id]". Default is None.'))
@click.option('--stdout_path', '-sout', default=None, type=STR,
              help='Subprocess standard output path. Default is in out_dir.')
@click.pass_context
def eagle(ctx, alloc, memory, walltime, feature, stdout_path):
    """Eagle submission tool for PLEXOS aggregation."""

    name = ctx.obj['NAME']
    job_input = ctx.obj['JOB_INPUT']
    out_dir = ctx.obj['OUT_DIR']
    reeds_dir = ctx.obj['REEDS_DIR']
    cf_years = ctx.obj['CF_YEARS']
    build_years = ctx.obj['BUILD_YEARS']
    scenario = ctx.obj['SCENARIO']
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
                cmd = get_node_cmd(node_name, job_input, out_dir, reeds_dir,
                                   cf_year, build_year, scenario, verbose)

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
