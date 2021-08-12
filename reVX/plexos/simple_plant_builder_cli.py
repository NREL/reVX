# -*- coding: utf-8 -*-
"""
Simple Plant Builder command line interface (cli).
"""
import os
import click
import logging

from rex.utilities.loggers import init_mult

from reVX.plexos.simple_plant_builder import SimplePlantBuilder
from reVX import __version__

logger = logging.getLogger(__name__)


@click.command()
@click.version_option(version=__version__)
@click.option('--plant_meta', '-pm', required=True,
              type=click.Path(exists=True),
              help=("Str filepath or extracted dataframe for plant meta data "
                    "with every row representing a plant with columns for "
                    "latitude, longitude, and capacity (in MW). Plants will "
                    "compete for available capacity in the reV supply curve "
                    "input and will be prioritized based on the row order of "
                    "this input."))
@click.option('--rev_sc', '-sc', required=True,
              type=click.Path(exists=True),
              help=("reV supply curve or sc-aggregation output table "
                    "including sc_gid, latitude, longitude, res_gids, "
                    "gid_counts, mean_cf."))
@click.option('--cf_fpath', '-cf', required=True,
              type=click.Path(exists=True),
              help=("File path to capacity factor file (reV gen output) to "
                    "get profiles from."))
@click.option('--out_fpath', '-o', required=True, type=click.Path(),
              help='Path to .h5 file to save plant data to')
@click.option('--forecast_fpath', '-fcst', default=None, type=click.Path(),
              show_default=True,
              help=("Forecasted capacity factor .h5 file path (reV results). "
                    "If not None, the generation profiles are sourced from "
                    "this file."))
@click.option('--no_share_resource', '-nsr', is_flag=True,
              help=("Flag to not share available capacity within a single "
                    "resource GID between multiple plants."))
@click.option('--max_workers', '-mw', default=None, type=int,
              help=("Max workers for parallel profile aggregation. None uses "
                    "all available workers. 1 will run in serial."))
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
def main(plant_meta, rev_sc, cf_fpath, out_fpath, forecast_fpath,
         no_share_resource, max_workers, verbose):
    """Simple Plant Builder Command Line Interface"""
    log_modules = [__name__, 'reVX', 'reV', 'rex']
    log_dir, log_name = os.path.split(out_fpath)
    init_mult(log_name.strip('.')[0], log_dir, modules=log_modules,
              verbose=verbose)
    logger.info('Running Simple Plant Builder specified in: {}'
                .format(plant_meta))
    logger.info('Plants to be built from reV sc table: {} and cf profiles: {}'
                .format(rev_sc, cf_fpath))
    logger.info('Outputs to be stored save to: {}'.format(out_fpath))
    SimplePlantBuilder.run(plant_meta, rev_sc, cf_fpath, out_fpath=out_fpath,
                           forecast_fpath=forecast_fpath,
                           share_resource=no_share_resource,
                           max_workers=max_workers)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running Simple Plant Builder CLI')
        raise
