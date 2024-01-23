"""
Combine wet and dry costs.
"""
import json
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt

from reVX.least_cost_xmission.masks import Masks
from reVX.least_cost_xmission.config.constants import COMBINED_COSTS_H5_LAYER,\
    COMBINED_COSTS_TIFF, DEFAULT_DTYPE, WET_COSTS_TIFF
from reVX.least_cost_xmission.trans_layer_io_handler import TransLayerIoHandler

logger = logging.getLogger(__name__)


class CostCombiner:
    """
    Combine wet and dry costs.
    """
    def __init__(self, io_handler: TransLayerIoHandler, masks: Masks):
        self._io_handler = io_handler
        self._masks = masks

    def load_wet_costs(self, f_name = WET_COSTS_TIFF) -> npt.NDArray:
        """
        Load wet costs from file

        Parameters
        ----------
        f_name, optional
            Filename for wet costs GeoTIFF, by default WET_COSTS_TIFF

        Returns
        -------
            Wet costs array
        """
        if not Path(f_name).exists():
            raise IOError(f'Wet costs GeoTIFF {f_name} does not exist')
        logger.debug(f'Loading wet costs from {f_name}')
        return self._io_handler.load_tiff(f_name)

    def load_legacy_dry_costs(self, h5_fpath, layer_name) -> npt.NDArray:
        """
        Load legacy dry costs from H5 and reproject if necessary

        Parameters
        ----------
        h5_fpath
            H5 file with dry costs
        layer_name
            Name of costs layer

        Returns
        -------
            Array of costs
        """
        if not Path(h5_fpath).exists():
            raise IOError(f'H5 file {h5_fpath} does not exist')

        logger.debug(f'Loading dry costs layer {layer_name} from {h5_fpath}')
        costs = self._io_handler.load_h5_layer(layer_name, h5_fpath)
        if costs.shape == self._io_handler.shape:
            return costs

        # Dry costs have a different shape. Attempt to reproject
        logger.debug('Dry costs have an unexpected shape. Reprojecting.')
        attrs = self._io_handler.load_h5_attrs(layer_name, h5_fpath)
        json_profile = attrs['profile']
        profile = json.loads(json_profile)
        reprojected = self._io_handler.reproject(costs, profile, init_dest=-1)

        return reprojected

    def combine_costs(self, wet_costs: npt.NDArray,
                      dry_costs: npt.NDArray, landfall_cost: float,
                      save_tiff: bool = True):
        """
        Combine wet, dry, and landfall costs using appropriate masks

        Parameters
        ----------
        wet_costs
            Wet costs array
        dry_costs
            Dry costs array
        landfall_cost
            Cost to apply to landfall cells for account for conversion from
            underwater cables to land based transmission.
        save_tiff, optional
            Save combined costs to GeoTIFF if True, by default True
        """
        if wet_costs.shape != self._io_handler.shape:
            raise ValueError(
                f'Wet costs shape {wet_costs.shape} does not match shape '
                f' of template raster {self._io_handler.shape}'
            )
        if dry_costs.shape != self._io_handler.shape:
            raise ValueError(
                f'Dry costs shape {dry_costs.shape} does not match shape '
                f' of template raster {self._io_handler.shape}'
            )

        combined = np.zeros(self._io_handler.shape, dtype=DEFAULT_DTYPE)
        combined[self._masks.wet_mask] = wet_costs[self._masks.wet_mask]
        combined[self._masks.dry_mask] = dry_costs[self._masks.dry_mask]

        # TODO - include half of wet and half of dry cost in landfall cost
        combined[self._masks.landfall_mask] = landfall_cost

        if save_tiff:
            logger.debug('Saving combined costs to GeoTIFF')
            self._io_handler.save_tiff(combined, COMBINED_COSTS_TIFF)

        num_zeros = (combined == 0).sum()
        if num_zeros > 0:
            logger.warning(
                f'{num_zeros} occurrences of 0 are in the combined costs. '
                'This may cause erroneous paths and costs.'
            )

        logger.debug('Writing combined costs to H5')
        self._io_handler.write_to_h5(combined, COMBINED_COSTS_H5_LAYER)
