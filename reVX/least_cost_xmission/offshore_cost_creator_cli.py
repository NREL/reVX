# -*- coding: utf-8 -*-
# pylint: disable=all
"""
Xmission Cost Creator Command Line Interface
"""
import os
import sys
import click
import logging
from typing import Dict, Union

from rex.utilities.loggers import init_logger

from reVX import __version__
from reVX.least_cost_xmission.offshore_utilities import CombineRasters
from reVX.config.least_cost_xmission import OffshoreCreatorConfig,\
      BarrierFiles, FrictionFiles

KwargsDict = Dict[str,
                  Union[str, int, float, bool, BarrierFiles, FrictionFiles]]

logger = logging.getLogger(__name__)
init_logger('reVX', log_level="DEBUG")

# TODO - create land mask command


@click.command
@click.option('-c', '--config', 'config_fpath', type=click.Path(exists=True),
              required=True, help='Configuration JSON.')
@click.option('--create-h5/--dont-create-h5', default=True,
              help='Create a new offshore H5 data file.')
def from_config(config_fpath: str, create_h5: bool):
    """
    Create offshore barriers and frictions from a config file.
    """
    config = OffshoreCreatorConfig(config_fpath)
    if create_h5 and config.ex_offshore_h5_fpath is None:
        click.echo('ex_offshore_h5_fpath must be set unless H5 creation is '
                   'disabled with --dont-create-h5', err=True)
        sys.exit(1)

    kwargs: KwargsDict= {}
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

    kwargs = {}
    if config.slope_fpath is not None:
        kwargs['slope_file'] = config.slope_fpath
    if config.save_tiff is not None:
        kwargs['save_tiff'] = config.save_tiff
    cr.build_off_shore_barriers(config.barrier_files,
                                config.forced_inclusion_files, **kwargs)

    kwargs = {}
    if config.slope_fpath is not None:
        kwargs['slope_file'] = config.slope_fpath
    if config.save_tiff is not None:
        kwargs['save_tiff'] = config.save_tiff
    if config.bathy_fpath is not None:
        kwargs['bathy_file'] = config.bathy_fpath
    if config.bathy_depth_cutoff is not None:
        kwargs['bathy_depth_cutoff'] = config.bathy_depth_cutoff
    if config.bathy_friction is not None:
        kwargs['bathy_friction'] = config.bathy_friction
    if config.minimum_friction_files is not None:
        kwargs['minimum_friction_files'] = config.minimum_friction_files
    cr.build_off_shore_friction(config.friction_files, **kwargs)

    if create_h5:
        cr.create_offshore_h5(config.ex_offshore_h5_fpath,
                              config.offshore_h5_fpath,
                              overwrite=config.overwrite_h5)

    cr.load_land_mask(mask_f=config.land_mask_fpath)

    kwargs = {}
    if config.save_tiff is not None:
        kwargs['save_tiff'] = config.save_tiff
    cr.merge_os_and_land_barriers(config.land_h5_fpath,
                                  config.land_barrier_layer,
                                  config.offshore_h5_fpath, **kwargs)

    kwargs = {}
    if config.save_tiff is not None:
        kwargs['save_tiff'] = config.save_tiff
    if config.land_cost_mult is not None:
        kwargs['land_cost_mult'] = config.land_cost_mult
    cr.merge_os_and_land_friction(config.land_h5_fpath,
                                  config.land_costs_layer,
                                  config.offshore_h5_fpath,
                                  **kwargs)

if __name__ == '__main__':
    from_config()  # pylint: disable=no-value-for-parameter