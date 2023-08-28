# -*- coding: utf-8 -*-
# pylint: disable=all
"""
Offshore Xmission Friction and Barrier Creator Command Line Interface
"""
import sys
import click
import logging
from typing import Dict, Union

from rex.utilities.loggers import init_logger, LOG_LEVEL

from reVX import __version__
from reVX.least_cost_xmission.offshore_utilities import CombineRasters
from reVX.least_cost_xmission.offshore_utilities import convert_pois_to_lines
from reVX.config.least_cost_xmission import OffshoreCreatorConfig

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--log-level', type=click.Choice(LOG_LEVEL.keys()),
              default='INFO', help='Logging level.')
def main(log_level):
    """
    Offshore Layer Creator Command Line Interface
    """
    init_logger('reVX', log_level=log_level)


@main.command
@click.option('--vector', '-v', required=True, type=click.Path(exists=True),
              help='Vector land mask to rasterize.')
@click.option('--template-raster', '-t', required=True,
              type=click.Path(exists=True),
              help='Raster to use as a template for rasterizing the land mask')
@click.option('--out-file', '-o', type=click.Path(),
              help='Filename to use for rasterized land mask.')
@click.option('--buffer', '-b', type=int,
              help='Buffer vector features before rasterizing.')
def rasterize_land_mask(vector: str, template_raster: str,
                        out_file: Union[str, None],
                        buffer: Union[int, None]):
    """
    Convert a vector land mask to a raster based on a template. Any features in
    the vector layer will be assigned a 1 value in the raster, all other cells
    will have a value of 0.
    """
    cr = CombineRasters(template_raster)
    cr.create_land_mask(vector, save_tiff=True, filename=out_file,
                        buffer_dist=buffer)


@main.command
@click.option('--poi-file', '-p', required=True, type=click.Path(exists=True),
              help='File of POIs in CSV format. Each POI must have the '
              'following fields: "POI Name", "State", "Voltage (kV)", "Lat", '
              'and "Long". "State" may be blank. Other fields are ignored.')
@click.option('--template-raster', '-t', required=True,
              type=click.Path(exists=True),
              help='Raster to extract CRS from.')
@click.option('--out-file', '-o', type=click.Path(), required=True,
              help='Filename to use for POI lines GeoPackage file.')
def convert_pois(poi_file: str, template_raster: str, out_file: str):
    """
    Convert points of interconnection (POI) to short lines. The transmission
    routing code requires all transmission elemnts to be lines. The POIs
    defined in the CSV will be converted to lines and labeled as substations.
    As all substations must be link to a transmission line, a synthetic
    transmission line is created that is linked to the POIs.
    """
    convert_pois_to_lines(poi_file, template_raster, out_file)


@main.command
@click.option('-c', '--config', 'config_fpath', type=click.Path(exists=True),
              required=True, help='Configuration JSON.')
@click.option('--create-h5/--use-existing-h5', default=True, show_default=True,
              help='Create a new offshore H5 data file or use an existing H5.')
def from_config(config_fpath: str, create_h5: bool):
    """
    Create offshore barriers and frictions from a config file.
    """
    config = OffshoreCreatorConfig(config_fpath)
    if create_h5 and config.ex_offshore_h5_fpath is None:
        click.echo('ex_offshore_h5_fpath must be set unless H5 creation is '
                   'disabled with --dont-create-h5', err=True)
        sys.exit(1)

    kwargs: Dict[str, Union[str, int, float]] = {}
    if config.layer_dir is not None:
        kwargs['layer_dir'] = config.layer_dir
    if config.slope_barrier_cutoff is not None:
        kwargs['slope_barrier_cutoff'] = config.slope_barrier_cutoff
    if config.low_slope_cutoff is not None:
        kwargs['low_slope_cutoff'] = config.low_slope_cutoff
    if config.high_slope_friction is not None:
        kwargs['high_slope_friction'] = config.high_slope_friction
    if config.medium_slope_friction is not None:
        kwargs['medium_slope_friction'] = config.medium_slope_friction
    if config.low_slope_friction is not None:
        kwargs['low_slope_friction'] = config.low_slope_friction
    cr = CombineRasters(config.template_raster_fpath, **kwargs)

    cr.build_off_shore_barriers(config.barrier_files,
                                config.forced_inclusion_files,
                                slope_file=config.slope_fpath,
                                save_tiff=config.save_tiff)

    min_fric_files = config.minimum_friction_files
    cr.build_off_shore_friction(config.friction_files,
                                slope_file=config.slope_fpath,
                                save_tiff=config.save_tiff,
                                bathy_file=config.bathy_fpath,
                                bathy_depth_cutoff=config.bathy_depth_cutoff,
                                bathy_friction=config.bathy_friction,
                                minimum_friction_files=min_fric_files)

    if create_h5:
        cr.create_offshore_h5(config.ex_offshore_h5_fpath,
                              config.offshore_h5_fpath,
                              overwrite=config.overwrite_h5)

    cr.load_land_mask(mask_f=config.land_mask_fpath)

    cr.merge_os_and_land_barriers(config.land_h5_fpath,
                                  config.land_barrier_layer,
                                  config.offshore_h5_fpath,
                                  save_tiff=config.save_tiff)

    kwargs = {}
    if config.land_cost_mult is not None:
        kwargs['land_cost_mult'] = config.land_cost_mult
    cr.merge_os_and_land_friction(config.land_h5_fpath,
                                  config.land_costs_layer,
                                  config.offshore_h5_fpath,
                                  save_tiff=config.save_tiff,
                                  **kwargs)


if __name__ == '__main__':
    try:
        main()  # pylint: disable=no-value-for-parameter
    except Exception:
        logger.exception('Error running Offshore Layer Creator CLI')
        raise
