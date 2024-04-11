"""
Combine wet and dry costs.
"""
import json
import logging
from pathlib import Path
from warnings import warn

import numpy as np
import numpy.typing as npt

from reVX.handlers.layered_h5 import LayeredTransmissionH5
from reVX.least_cost_xmission.layers.masks import Masks
from reVX.least_cost_xmission.config.constants import (
    COMBINED_COSTS_TIFF, DEFAULT_DTYPE, WET_COSTS_TIFF,
    COMBINED_COSTS_H5_LAYER_NAME, WET_COSTS_H5_LAYER_NAME,
    DRY_COSTS_H5_LAYER_NAME, LANDFALL_COSTS_H5_LAYER_NAME)

logger = logging.getLogger(__name__)


class CostCombiner:
    """
    Combine wet and dry costs.
    """
    def __init__(self, io_handler: LayeredTransmissionH5, masks: Masks):
        """
        Parameters
        ----------
        io_handler : :class:`LayeredTransmissionH5`
            Transmission layer handler.
        masks : Masks
            Masks instance.
        """
        self._io_handler = io_handler
        self._masks = masks

    def load_wet_costs(self, fname: str = WET_COSTS_TIFF) -> npt.NDArray:
        """
        Load wet costs from file

        Parameters
        ----------
        fname : str, optional
            Filename for wet costs GeoTIFF, by default WET_COSTS_TIFF

        Returns
        -------
        array-like
            Wet costs array
        """
        if not Path(fname).exists():
            raise FileNotFoundError(f'Wet costs GeoTIFF {fname} does not '
                                    'exist')
        logger.debug(f'Loading wet costs from {fname}')
        return self._io_handler.load_data_using_h5_profile(fname)

    def load_dry_costs(self, fname: str) -> npt.NDArray:
        """
        Load costs from file

        Parameters
        ----------
        fname : str
            Filename for dry costs GeoTIFF

        Returns
        -------
        array-like
            Dry costs array
        """
        if not Path(fname).exists():
            raise FileNotFoundError(f'Det costs GeoTIFF {fname} does not '
                                    'exist')
        logger.debug(f'Loading dry costs from {fname}')
        return self._io_handler.load_data_using_h5_profile(fname,
                                                           reproject=True)

    def combine_costs(self, wet_costs: npt.NDArray,
                      dry_costs: npt.NDArray, landfall_cost: float,
                      combined_layer_name: str = COMBINED_COSTS_H5_LAYER_NAME,
                      wet_layer_name: str = WET_COSTS_H5_LAYER_NAME,
                      dry_layer_name: str = DRY_COSTS_H5_LAYER_NAME,
                      landfall_layer_name: str = LANDFALL_COSTS_H5_LAYER_NAME,
                      output_tiff_dir = None):
        """
        Combine wet, dry, and landfall costs using appropriate masks. Write
        all layers to H5. Individual costs layers are set to zero outside of
        their domains before saving to H5.

        Parameters
        ----------
        wet_costs : array-like
            Wet costs array
        dry_costs : array-like
            Dry costs array
        landfall_cost : float
            Cost to apply to landfall cells for conversion from underwater
            cables to land based transmission.
        combined_layer_name : str
            Name for combined costs in H5 file
        wet_layer_name : str
            Name for wet costs in H5 file
        dry_layer_name : str
            Name for dry costs in H5 file
        landfall_layer_name : str
            Name for landfall costs in H5 file
        output_tiff_dir : path-like, optional
            Path to output firectory to aave combined costs as GeoTIFF.
            If ``None``, combined costs are not saved.
            By default, ``None``.
        """
        if wet_costs.shape != self._io_handler.shape:
            raise ValueError(f'Wet costs shape {wet_costs.shape} does not '
                             'match shape of template raster '
                             f'{self._io_handler.shape}')

        if dry_costs.shape != self._io_handler.shape:
            raise ValueError(f'Dry costs shape {dry_costs.shape} does not '
                             'match shape of template raster '
                             f'{self._io_handler.shape}')

        combined = np.zeros(self._io_handler.shape, dtype=DEFAULT_DTYPE)
        landfall_costs = np.zeros(self._io_handler.shape, dtype=DEFAULT_DTYPE)

        landfall_costs[self._masks.landfall_mask] = landfall_cost
        combined[self._masks.wet_mask] = wet_costs[self._masks.wet_mask]
        combined[self._masks.dry_mask] = dry_costs[self._masks.dry_mask]
        combined[self._masks.landfall_mask] = \
            landfall_costs[self._masks.landfall_mask]

        if output_tiff_dir is not None:
            out_fp = Path(output_tiff_dir) / COMBINED_COSTS_TIFF
            logger.debug('Saving combined costs to GeoTIFF: %s', str(out_fp))
            self._io_handler.save_data_using_h5_profile(combined, out_fp)

        num_zeros = (combined == 0).sum()
        if num_zeros > 0:
            msg = (f'{num_zeros} occurrences of 0 are in the combined costs. '
                   'This may cause erroneous paths and costs.')
            logger.warning(msg)
            warn(msg)

        logger.debug('Writing combined costs to H5')
        self._io_handler.write_layer_to_h5(combined, combined_layer_name)

        logger.debug('Writing wet costs to H5')
        wet_costs = wet_costs.copy()
        wet_costs[~self._masks.wet_mask] = 0
        self._io_handler.write_layer_to_h5(wet_costs, wet_layer_name)

        logger.debug('Writing dry costs to H5')
        dry_costs = dry_costs.copy()
        dry_costs[~self._masks.dry_mask] = 0
        self._io_handler.write_layer_to_h5(dry_costs, dry_layer_name)

        logger.debug('Writing landfall costs to H5')
        self._io_handler.write_layer_to_h5(landfall_costs, landfall_layer_name)
