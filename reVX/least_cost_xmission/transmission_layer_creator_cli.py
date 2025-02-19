"""
CLI to create wet and dry costs, barriers, and friction layers. Final
layers required for LCP routing are saved to an H5 file. All layers may
optionally be saved to GeoTIFF.
"""
import sys
import click
import logging
from pathlib import Path
from warnings import warn

import numpy as np
from pydantic import ValidationError
from rex.utilities.loggers import init_mult
from gaps.config import load_config

from reVX import __version__
from reVX.config.transmission_layer_creation import (
    TransmissionLayerCreationConfig, MergeFrictionBarriers
)
from reVX.least_cost_xmission.config.constants import ALL
from reVX.handlers.layered_h5 import LayeredTransmissionH5
from reVX.least_cost_xmission.layers import LayerCreator
from reVX.least_cost_xmission.layers.masks import Masks
from reVX.least_cost_xmission.layers.utils import convert_pois_to_lines
from reVX.least_cost_xmission.layers.dry_cost_creator import DryCostCreator


logger = logging.getLogger(__name__)
CONFIG_ACTIONS = ['layers', 'dry_costs', 'merge_friction_and_barriers']


@click.group()
@click.version_option(version=__version__)
@click.option('--verbose', '-v', is_flag=True, default=False,
              help='Flag to turn on debug logging. Default is not verbose.')
def main(verbose):
    """
    Transmission LCP routing cost, friction, and barrier layer creator.
    """
    log_modules = [__name__, 'reVX', 'reV', 'rex']
    init_mult('trans-creator', None, modules=log_modules, verbose=verbose)


@main.command
@click.option('-c', '--config', 'config_fpath', type=click.Path(exists=True),
              required=True, help='Configuration JSON.')
def from_config(config_fpath: str):  # noqa: C901
    """
    Create costs, barriers, and frictions from a config file.
    """
    config_dict = load_config(config_fpath)
    try:
        config = TransmissionLayerCreationConfig.model_validate(config_dict)
    except ValidationError as e:
        logger.error(f'Error loading config file {config_fpath}:\n{e}')
        sys.exit(1)

    if not any(map(lambda key: config.model_dump()[key] is not None,
                   CONFIG_ACTIONS)):
        logger.error(f'At least one of {CONFIG_ACTIONS} must be in the '
                     'config file')
        sys.exit(1)

    # Done with guard clauses
    output_tiff_dir = Path(config.output_tiff_dir).expanduser().resolve()
    output_tiff_dir.mkdir(exist_ok=True, parents=True)
    template_file = str(config.template_raster_fpath)
    h5_io_handler = LayeredTransmissionH5(h5_file=str(config.h5_fpath),
                                          template_file=template_file,
                                          layer_dir=config.layer_dir)

    masks = _load_masks(config, h5_io_handler)

    # Perform actions in config
    builder = LayerCreator(h5_io_handler, masks, output_tiff_dir,
                           cell_size=config.cell_size)
    for lc in config.layers or []:
        builder.build(lc.layer_name, lc.build,
                      values_are_costs_per_mile=lc.values_are_costs_per_mile,
                      write_to_h5=lc.include_in_h5, description=lc.description)

    if config.dry_costs is not None:
        dc = config.dry_costs
        template_file = str(dc.iso_region_tiff)

        try:
            dry_mask = masks.dry_mask
        except ValueError:
            msg = "Dry mask not found! Computing dry costs for full extent!"
            logger.warning(msg)
            warn(msg)
            dry_mask = np.full(h5_io_handler.shape, True)

        dcc = DryCostCreator(h5_io_handler, dry_mask, output_tiff_dir,
                             cell_size=config.cell_size)
        cost_configs = None if not dc.cost_configs else str(dc.cost_configs)
        dcc.build(str(dc.iso_region_tiff), str(dc.nlcd_tiff),
                  str(dc.slope_tiff), cost_configs=cost_configs,
                  default_mults=dc.default_mults, extra_tiffs=dc.extra_tiffs)

    if config.merge_friction_and_barriers is not None:
        _combine_friction_and_barriers(config.merge_friction_and_barriers,
                                       h5_io_handler, output_tiff_dir)


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
    routing code requires all transmission elements to be lines. The POIs
    defined in the CSV will be converted to lines and labeled as substations.
    As all substations must be link to a transmission line, a synthetic
    transmission line is created that is linked to the POIs.
    """
    convert_pois_to_lines(poi_file, template_raster, out_file)


@main.command
@click.option('--land-mask-vector', '-l', required=True,
              type=click.Path(exists=True),
              help='GeoPackage or shapefile representing land.')
@click.option('--template-raster', '-t', required=True,
              type=click.Path(exists=True),
              help='Raster to extract CRS, transform, and shape from.')
@click.option('--masks-dir', '-m', type=click.Path(), required=False,
              default='.',
              help='Directory to store mask GeoTIFFs in.')
@click.option('--dont-reproject', '-d', is_flag=True, default=False,
              help='Don\'t reproject vector before creating masks.')
def create_masks(land_mask_vector: str, template_raster: str, masks_dir: str,
                 dont_reproject: bool):
    """
    Convert a vector land file to wet, dry, and all other masks.
    """
    io_handler = LayeredTransmissionH5(template_file=template_raster)
    masks = Masks(io_handler, masks_dir=masks_dir)

    reproject = not dont_reproject
    masks.create_masks(land_mask_vector, save_tiff=True,
                       reproject_vector=reproject)


@main.command
@click.option('--template-raster', '-t', type=click.Path(exists=True),
              required=True,
              help='Raster to extract CRS, transform, and shape from.')
@click.option('--h5-file', '-h', type=click.Path(exists=False), required=True,
              help='Name of H5 file to create.')
@click.option('--block_size', '-bs', type=int, required=False, default=None,
              help='Block size used to build lat/lon datasets.')
def create_h5(template_raster: str, h5_file: str, block_size: int):
    """
    Create a new H5 file to store layers in.
    """
    logger.info('Using raster %s to create new H5 file %s', template_raster,
                h5_file)
    lth5 = LayeredTransmissionH5(h5_file, template_file=template_raster,
                                 block_size=block_size)
    lth5.create_new()


def _load_masks(config, h5_io_handler):
    """Load masks based on config file.

    Parameters
    ----------
    config : :class:`MergeFrictionBarriers`
        Config object
    h5_io_handler : :class:`LayeredTransmissionH5`
        Transmission IO handler

    Returns
    -------
    Masks
        Masks instance based on directories in config file.
    """
    masks = Masks(h5_io_handler, masks_dir=config.masks_dir)
    if not config.layers:
        return masks

    build_configs = [lc.build for lc in config.layers]
    need_masks = any(lc.extent != ALL
                     for bc in build_configs for lc in bc.values())
    if need_masks:
        masks.load_masks()

    return masks


def _combine_friction_and_barriers(config: MergeFrictionBarriers,
                                   io_handler: LayeredTransmissionH5,
                                   output_tiff_dir=None):
    """
    Combine friction and barriers and save to H5 and optionally GeoTIFF

    Parameters
    ----------
    config : :class:`MergeFrictionBarriers`
        Config object
    io_handler : :class:`LayeredTransmissionH5`
        Transmission IO handler
    output_tiff_dir : path-like, optional
        Directory where combined barriers should be saved as GeoTIFF. If
        ``None``, combined layers are not saved. By default, ``None``.
    """
    output_tiff_dir = Path(output_tiff_dir)
    friction_tiff = output_tiff_dir / f"{config.friction_layer}.tif"
    raw_barrier_tiff = output_tiff_dir / f"{config.barrier_layer}.tif"
    if not friction_tiff.exists():
        logger.error(f'The friction GeoTIFF ({str(friction_tiff)}) was not '
                     'found. Please create it using the `friction_layers` '
                     'key in the config file.')
        sys.exit(1)

    if not raw_barrier_tiff.exists():
        logger.error(f'The raw barriers GeoTIFF ({str(raw_barrier_tiff)}) was '
                     'not found. Please create it using the `barrier_layers` '
                     'key in the config file.')
        sys.exit(1)

    logger.info('Loading friction and raw barriers.')
    friction = io_handler.load_data_using_h5_profile(friction_tiff)
    barriers = io_handler.load_data_using_h5_profile(raw_barrier_tiff)

    combined = friction + barriers * config.barrier_multiplier

    if output_tiff_dir is not None:
        out_fp = output_tiff_dir / f"{config.output_layer_name}.tif"
        logger.debug('Saving combined barriers to %s', out_fp)
        io_handler.save_data_using_h5_profile(combined, out_fp)

    logger.info('Writing combined barriers to H5')
    io_handler.write_layer_to_h5(combined, config.output_layer_name)


if __name__ == '__main__':
    try:
        main()  # pylint: disable=no-value-for-parameter
    except Exception:
        logger.exception('Error running Offshore Layer Creator CLI')
        raise
