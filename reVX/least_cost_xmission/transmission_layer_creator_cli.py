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
from reVX.least_cost_xmission.masks import Masks
from reVX.least_cost_xmission.json_config import LayerCreationConfig
from reVX.least_cost_xmission.offshore_cost_creator import OffshoreCostCreator
from reVX.least_cost_xmission.trans_layer_io_handler import TransLayerIoHandler
from reVX.least_cost_xmission.friction_barrier_builder import FrictionBarrierBuilder

logger = logging.getLogger(__name__)

CONFIG_ACTIONS = [
    'friction_layers', 'barrier_layers', 'wet_costs', 'dry_costs',
    'combine_costs'
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
@click.option('--create-h5/--use-existing-h5', default=True, show_default=True,
              help='Create a new H5 data file or use an existing H5.')
def from_config(config_fpath: str, create_h5: bool):
    """
    Create costs, barriers, and frictions from a config file.
    """
    with open(config_fpath, 'r') as inf:
        raw_json = inf.read()
    try:
        config = LayerCreationConfig.model_validate_json(raw_json)
    except ValidationError as e:
        click.echo(f'Error loading config file {config_fpath}:\n{e}')
        sys.exit(1)

    if not any(map(lambda key: config.model_dump()[key] is not None,
                   CONFIG_ACTIONS)):
        click.echo(
            f'At least one of {CONFIG_ACTIONS} must be in the config file'
        )
        sys.exit(1)

    save_tiff = config.save_tiff

    # Done with guard clauses
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
        fbb.build_layer(config.barrier_layers, save_tiff=save_tiff)

    if config.friction_layers is not None:
        fbb = FrictionBarrierBuilder('friction', io_handler, masks)
        fbb.build_layer(config.friction_layers, save_tiff=save_tiff)

    if config.wet_costs is not None:
        wc = config.wet_costs
        occ = OffshoreCostCreator(io_handler)
        if wc.wet_costs_tiff is None:
            occ.build_offshore_costs(str(wc.bathy_tiff), wc.bins)
        else:
            occ.build_offshore_costs(
                str(wc.bathy_tiff), wc.bins, str(wc.wet_costs_tiff)
            )

    if config.dry_costs is not None:
        # TODO
        raise NotImplementedError(
            'The "dry_costs" option is not supported yet'
        )

    if config.combine_costs is not None:
        pass


def _setup_h5_files(io_handler: TransLayerIoHandler,
                    h5_fpath: Path, existing_h5_fpath: Optional[Path]):
    """TODO

    Parameters
    ----------
    io_handler
        _description_
    h5_fpath
        _description_
    existing_h5_fpath
        _description_
    """
    if h5_fpath.exists():
        click.echo(f'Using H5 file {h5_fpath} for storing data.')
        io_handler.set_h5_file(str(h5_fpath))
        return

    # h5_fpath doesn't exist, create it
    if existing_h5_fpath is None:
        click.echo(
            f'"h5_fpath" {h5_fpath} does not exist. "existing_h5_fpath" '
            'must be set to an existing H5 file.'
        )
        sys.exit(1)

    if not existing_h5_fpath.exists():
        click.echo(
            f'"existing_h5_fpath" {existing_h5_fpath} does not exist, and '
            'is required to create a new H5.'
        )
        sys.exit(1)

    click.echo(f'Creating new H5 {h5_fpath} with meta data from '
                f'{existing_h5_fpath}')
    io_handler.create_new_h5(str(existing_h5_fpath), str(h5_fpath))


if __name__ == '__main__':
    try:
        main()  # pylint: disable=no-value-for-parameter
    except Exception:
        logger.exception('Error running Offshore Layer Creator CLI')
        raise