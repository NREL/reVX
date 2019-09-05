# -*- coding: utf-8 -*-
"""
PLEXOS command line interface (cli).
"""
import os
import pandas as pd
import click
import logging
from reV.utilities.execution import SLURM
from reV.utilities.cli_dtypes import STR, INT, INTLIST
from reV.utilities.loggers import init_mult
from reX.plexos.node_aggregation import Manager

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
@click.option('--cf_year', '-y', required=True, type=INT,
              help='Capacity factor resource year.')
@click.option('--build_years', '-by', required=True, type=INTLIST,
              help='REEDS build years to aggregate profiles for.')
@click.option('--scenario', '-s', required=False, default=None, type=STR,
              help='Optional filter to run just one scenario from job input.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, job_input, out_dir, reeds_dir, cf_year, build_years,
         scenario, verbose):
    """PLEXOS Command Line Interface"""

    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['JOB_INPUT'] = job_input
    ctx.obj['OUT_DIR'] = out_dir
    ctx.obj['REEDS_DIR'] = reeds_dir
    ctx.obj['CF_YEAR'] = cf_year
    ctx.obj['BUILD_YEARS'] = build_years
    ctx.obj['SCENARIO'] = scenario
    ctx.obj['VERBOSE'] = verbose

    if ctx.invoked_subcommand is None:
        init_mult(name, out_dir, modules=[__name__, 'reX.plexos'],
                  verbose=verbose)
        logger.info('Running reV to PLEXOS pipeline using job input:\n{}'
                    '\nOutputs to be stored in:\n{}'
                    .format(job_input, out_dir))
        logger.info('Aggregating plexos scenario "{}".'.format(scenario))
        Manager.run(job_input, out_dir, reeds_dir, scenario=scenario,
                    cf_year=cf_year, build_years=build_years)


def get_node_cmd(name, job_input, out_dir, reeds_dir, cf_year, build_years,
                 scenario, verbose):
    """Get a CLI call command for the plexos CLI."""

    args = ('-n {name} '
            '-j {job} '
            '-o {out} '
            '-rd {reeds} '
            '-y {year} '
            '-by {build} '
            '-s {scenario} ')

    args = args.format(name=SLURM.s(name),
                       job=SLURM.s(job_input),
                       out=SLURM.s(out_dir),
                       reeds=SLURM.s(reeds_dir),
                       year=SLURM.s(cf_year),
                       build=SLURM.s(build_years),
                       scenario=SLURM.s(scenario))

    if verbose:
        args += '-v '

    cmd = 'python -m reX.plexos.plexos_cli {}'.format(args)
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
    cf_year = ctx.obj['CF_YEAR']
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

    for scenario in scenarios:
        for year in build_years:
            node_name = '{}_{}_{}'.format(name, scenario.replace(' ', '_'),
                                          year)
            cmd = get_node_cmd(node_name, job_input, out_dir, reeds_dir,
                               cf_year, [year], scenario, verbose)

            logger.info('Running reX plexos aggregation on Eagle with node '
                        'name "{}"'.format(node_name))

            slurm = SLURM(cmd, alloc=alloc, memory=memory, walltime=walltime,
                          feature=feature, name=node_name,
                          stdout_path=stdout_path)
            if slurm.id:
                msg = ('Kicked off reX plexos aggregation job "{}" '
                       '(SLURM jobid #{}) on Eagle.'
                       .format(node_name, slurm.id))
            else:
                msg = ('Was unable to kick off reV generation job "{}". '
                       'Please see the stdout error messages'
                       .format(node_name))
            click.echo(msg)
            logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running PLEXOS CLI')
        raise
