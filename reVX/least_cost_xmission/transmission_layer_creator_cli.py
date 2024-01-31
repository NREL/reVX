"""
CLI to create wet (TODO and dry) costs, barriers, and friction layers. Final
layers required for LCP routing are saved to an H5 file. All layers may
optionally be saved to GeoTIFF.
"""
import sys
from typing import Optional
import click
import logging
from pathlib import Path

from pydantic import ValidationError
from rex.utilities.loggers import init_mult

from reVX import __version__
from reVX.config.transmission_layer_creation import LayerCreationConfig, \
    MergeFrictionBarriers
from reVX.least_cost_xmission.config.constants import BARRIER_H5_LAYER_NAME, \
    BARRIER_TIFF, FRICTION_TIFF, RAW_BARRIER_TIFF

from reVX.least_cost_xmission.layers.masks import Masks
from reVX.least_cost_xmission.costs.cost_combiner import CostCombiner
from reVX.least_cost_xmission.costs.wet_cost_creator import WetCostCreator
from reVX.least_cost_xmission.layers.friction_barrier_builder import \
    FrictionBarrierBuilder
from reVX.least_cost_xmission.layers.transmission_layer_io_handler import \
    TransLayerIoHandler
from reVX.least_cost_xmission.layers.utils import convert_pois_to_lines

logger = logging.getLogger(__name__)

CONFIG_ACTIONS = [
    'friction_layers', 'barrier_layers', 'wet_costs', 'dry_costs',
    'combine_costs', 'merge_friction_and_barriers',
]


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
    with open(config_fpath, 'r') as inf:
        raw_json = inf.read()
    try:
        config = LayerCreationConfig.model_validate_json(raw_json)
    except ValidationError as e:
        logger.error(f'Error loading config file {config_fpath}:\n{e}')
        sys.exit(1)

    if not any(map(lambda key: config.model_dump()[key] is not None,
                   CONFIG_ACTIONS)):
        logger.error(
            f'At least one of {CONFIG_ACTIONS} must be in the config file'
        )
        sys.exit(1)

    # Done with guard clauses
    save_tiff = config.save_tiff
    io_handler = TransLayerIoHandler(str(config.template_raster_fpath),
                                     layer_dir=config.layer_dir)

    _setup_h5_files(io_handler, config.h5_fpath, config.existing_h5_fpath)

    masks = Masks(io_handler, masks_dir=config.masks_dir)
    if config.land_mask_vector_fname is not None:
        masks.create_masks(config.land_mask_vector_fname, save_tiff=save_tiff)
    else:
        masks.load_masks()

    # Perform actions in config
    if config.barrier_layers is not None:
        fbb = FrictionBarrierBuilder('barrier', io_handler, masks)
        fbb.build_layer(config.barrier_layers)

    if config.friction_layers is not None:
        fbb = FrictionBarrierBuilder('friction', io_handler, masks)
        fbb.build_layer(config.friction_layers)

    if config.wet_costs is not None:
        wc = config.wet_costs
        wcc = WetCostCreator(io_handler)
        if wc.wet_costs_tiff is None:
            wcc.build_wet_costs(str(wc.bathy_tiff), wc.bins)
        else:
            wcc.build_wet_costs(
                str(wc.bathy_tiff), wc.bins, str(wc.wet_costs_tiff)
            )

    if config.dry_costs is not None:
        # TODO - implement this
        raise NotImplementedError(
            'The "dry_costs" option is not supported yet'
        )

    if config.merge_friction_and_barriers is not None:
        combine_friction_and_barriers(config.merge_friction_and_barriers,
                                      io_handler, save_tiff=save_tiff)

    if config.combine_costs is not None:
        cc = config.combine_costs
        combiner = CostCombiner(io_handler, masks)
        wet_costs = combiner.load_wet_costs()
        dry_costs = combiner.load_legacy_dry_costs(cc.dry_h5_fpath,
                                                   cc.dry_costs_layer)
        combiner.combine_costs(wet_costs, dry_costs, cc.landfall_cost,
                               cc.dry_costs_layer, save_tiff=save_tiff)


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


def _setup_h5_files(io_handler: TransLayerIoHandler,
                    h5_fpath: Path, existing_h5_fpath: Optional[Path]):
    """
    Load or Create new H5 file as needed

    Parameters
    ----------
    io_handler
        The transmission IO handler
    h5_fpath
        Path to H5 file to use for current analysis
    existing_h5_fpath
        Existing H5 file to pull meta data from
    """
    if h5_fpath.exists():
        logger.info(f'Using H5 file {h5_fpath} for storing data.')
        io_handler.set_h5_file(str(h5_fpath))
        return

    # h5_fpath doesn't exist, create it
    if existing_h5_fpath is None:
        logger.error(
            f'"h5_fpath" {h5_fpath} does not exist. "existing_h5_fpath" '
            'must be set to an existing H5 file.'
        )
        sys.exit(1)

    if not existing_h5_fpath.exists():
        logger.error(
            f'"existing_h5_fpath" {existing_h5_fpath} does not exist, and '
            'is required to create a new H5.'
        )
        sys.exit(1)

    logger.info(f'Creating new H5 {h5_fpath} with meta data from '
                f'{existing_h5_fpath}')
    io_handler.create_new_h5(str(existing_h5_fpath), str(h5_fpath))


def combine_friction_and_barriers(config: MergeFrictionBarriers,
                                  io_handler: TransLayerIoHandler,
                                  save_tiff: bool = True):
    """
    Combine friction and barriers and save to H5 and optionally GeoTIFF

    Parameters
    ----------
    config
        Config object
    io_handler
        Transmission IO handler
    save_tiff
        Save combined barriers to GeoTIFF if True
    """
    if not Path(FRICTION_TIFF).exists():
        logger.error(
            f'The friction GeoTIFF ({FRICTION_TIFF}) was not found. Please '
            'create it using the `friction_layers` key in the config file.'
        )

    if not Path(RAW_BARRIER_TIFF).exists():
        logger.error(
            f'The raw barriers GeoTIFF ({RAW_BARRIER_TIFF}) was not found. '
            'Please create it using the `barrier_layers` key in the config '
            'file.'
        )

    logger.info('Loading friction and raw barriers.')
    friction = io_handler.load_tiff(FRICTION_TIFF)
    barriers = io_handler.load_tiff(RAW_BARRIER_TIFF)

    combined = friction + barriers * config.barrier_multiplier

    if save_tiff:
        logger.debug('Saving combined barriers to GeoTIFF')
        io_handler.save_tiff(combined, BARRIER_TIFF)

    logger.info('Writing combined barriers to H5')
    io_handler.write_to_h5(combined, BARRIER_H5_LAYER_NAME)


if __name__ == '__main__':
    try:
        main()  # pylint: disable=no-value-for-parameter
    except Exception:
        logger.exception('Error running Offshore Layer Creator CLI')
        raise
