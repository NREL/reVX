# -*- coding: utf-8 -*-
"""
ResourceX Command Line Interface
"""
import click
import logging
import os

from reVX.utilities.loggers import init_mult
from reVX.resource.resource import ResourceX

logger = logging.getLogger(__name__)


@click.group()
@click.option('--resource_h5', '-h5', required=True,
              type=click.Path(exists=True),
              help=('Path to Resource .h5 file'))
@click.option('--out_dir', '-o', required=True, type=click.Path(),
              help='Directory to dump output files')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, resource_h5, out_dir, verbose):
    """
    ResourceX Command Line Interface
    """
    ctx.ensure_object(dict)
    ctx.obj['H5'] = resource_h5
    ctx.obj['OUT_DIR'] = out_dir
    ctx.obj['CLS'] = ResourceX

    name = os.path.splitext(os.path.basename(resource_h5))[0]
    init_mult(name, out_dir, verbose=verbose, node=True,
              modules=[__name__, 'reVX.resource.resource',
                       'reV.handlers.resource'])

    logger.info('Extracting Resource data from {}'.format(resource_h5))
    logger.info('Outputs to be stored in: {}'.format(out_dir))


@main.command()
@click.option('--lat_lon', '-ll', nargs=2, type=click.Tuple([float, float]),
              required=True, help='(lat, lon) coordinates of interest')
@click.pass_context
def sam(ctx, lat_lon):
    """
    Extract all datasets needed for SAM for the nearest pixel to the given
    (lat, lon) coordinates
    """
    with ctx.obj['CLS'](ctx.obj['H5']) as f:
        SAM_df = f.get_SAM_df(lat_lon)

    out_path = "{}.csv".format(SAM_df.name)
    out_path = os.path.join(ctx.obj['OUT_DIR'], out_path)
    logger.info('Saving data to {}'.format(out_path))
    SAM_df.to_csv(out_path)


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
    with ctx.obj['CLS'](ctx.obj['H5']) as f:
        site_df = f.get_site_df(dataset, lat_lon)

    gid = site_df.name
    out_path = "{}-{}.csv".format(dataset, gid)
    out_path = os.path.join(ctx.obj['OUT_DIR'], out_path)
    logger.info('Saving data to {}'.format(out_path))
    site_df.to_csv(out_path)


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
    with ctx.obj['CLS'](ctx.obj['H5']) as f:
        region_df = f.get_region_df(dataset, region, region_col=region_col)
        meta = f['meta']

    out_path = "{}-{}.csv".format(dataset, region)
    out_path = os.path.join(ctx.obj['OUT_DIR'], out_path)
    logger.info('Saving data to {}'.format(out_path))
    region_df.to_csv(out_path)

    out_path = "{}-meta.csv".format(region)
    out_path = os.path.join(ctx.obj['OUT_DIR'], out_path)
    meta = meta.loc[region_df.columns]
    logger.info('Saving meta data to {}'.format(out_path))
    meta.to_csv(out_path, index=False)


@main.command()
@click.option('--timestep', '-ts', type=str, required=True,
              help='Timestep to extract')
@click.option('--dataset', '-d', type=str, required=True,
              help='Dataset to extract')
@click.option('--region_col', '-col', type=str, default='state',
              help='Meta column to search for region')
@click.option('--region', '-r', type=str, default=None,
              help='Region to extract')
@click.pass_context
def map(ctx, timestep, dataset, region_col, region):
    """
    Extract a single dataset for a single timestep
    Extract only pixels in region if given.
    """
    with ctx.obj['CLS'](ctx.obj['H5']) as f:
        map_df = f.get_timestep_map(dataset, timestep, region=region,
                                    region_col=region_col)

    out_path = "{}-{}.csv".format(dataset, timestep)
    out_path = os.path.join(ctx.obj['OUT_DIR'], out_path)
    logger.info('Saving data to {}'.format(out_path))
    map_df.to_csv(out_path)


if __name__ == '__main__':
    main(obj={})
