# -*- coding: utf-8 -*-
"""
SolarX Command Line Interface
"""
import click
import logging
import os

from reVX.utilities.loggers import init_mult
from reVX.resource.resource import NSRDBX, SolarX
from reVX.resource.resource_cli import region as region_cmd
from reVX.resource.resource_cli import SAM as SAM_cmd
from reVX.resource.resource_cli import site as site_cmd

logger = logging.getLogger(__name__)


@click.group()
@click.option('--solar_h5', '-h5', required=True,
              type=click.Path(exists=True),
              help=('Path to Resource .h5 file'))
@click.option('--out_dir', '-o', required=True, type=click.Path(),
              help='Directory to dump output files')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, solar_h5, out_dir, verbose):
    """
    SolarX Command Line Interface
    """
    ctx.ensure_object(dict)
    ctx.obj['H5'] = solar_h5
    ctx.obj['OUT_DIR'] = out_dir

    name = os.path.splitext(os.path.basename(solar_h5))[0]
    if 'nsrdb' in name:
        ctx.obj['CLS'] = NSRDBX
    else:
        ctx.obj['CLS'] = SolarX

    init_mult(name, out_dir, verbose=verbose, node=True,
              modules=[__name__, 'reVX.resource', 'reV.handlers.resource'])

    logger.info('Extracting solar data from {}'.format(solar_h5))
    logger.info('Outputs to be stored in: {}'.format(out_dir))


@main.command()
@click.option('--lat_lon', '-ll', nargs=2, type=click.Tuple([float, float]),
              required=True, help='(lat, lon) coordinates of interest')
@click.pass_context
def SAM(ctx, lat_lon):
    """
    Extract all datasets needed for SAM for the nearest pixel to the given
    (lat, lon) coordinates
    """
    ctx.invoke(SAM_cmd, lat_lon=lat_lon)


@main.command()
@click.option('--dataset', '-d', type=str, required=True,
              help='Dataset to extract')
@click.option('--lat_lon', '-ll', nargs=2, type=click.Tuple([float, float]),
              required=True, help='(lat, lon) coordinates of interest')
@click.pass_context
def site(ctx, dataset, lat_lon):
    """
    Extract a single dataset for the nearest pixel to the given (lat, lon)
    coordinates
    """
    ctx.invoke(site_cmd, dataset=dataset, lat_lon=lat_lon)


@main.command()
@click.option('--dataset', '-d', type=str, required=True,
              help='Dataset to extract')
@click.option('--region', '-r', type=str, required=True,
              help='Region to extract')
@click.option('--region_col', '-col', type=str, default='state',
              help='Meta column to search for region')
@click.pass_context
def region(ctx, dataset, region, region_col):
    """
    Extract a single dataset for all pixels in the given region
    """
    ctx.invoke(region_cmd, dataset=dataset, region=region,
               region_col=region_col)


if __name__ == '__main__':
    main(obj={})
