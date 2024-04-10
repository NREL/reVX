"""
Create wet (offshore) costs and save to GeoTIFF.
"""
import logging
from typing import List
from warnings import warn

import numpy as np
import numpy.typing as npt
from reVX.least_cost_xmission.layers.masks import Masks
from reVX.config.transmission_layer_creation import RangeConfig
from reVX.least_cost_xmission.config.constants import (DEFAULT_DTYPE,
                                                       WET_COSTS_TIFF,
                                                       WET_COSTS_H5_LAYER_NAME)

from reVX.least_cost_xmission.layers.transmission_layer_io_handler import (
    TransLayerIoHandler
)

logger = logging.getLogger(__name__)


class WetCostCreator:
    """
    Create offshore costs and save to GeoTIFF.
    """
    def __init__(self, io_handler: TransLayerIoHandler, masks: Masks,
                 h5_io_handler=None):
        """
        Parameters
        ----------
        io_handler : TransLayerIoHandler
            Transmission layer IO handler
        masks : Masks
            Masks instance.
        h5_io_handler : TransLayerIoHandler, optional
            Optional H5 file handler. If provided, the cost layers will
            be saved to the H5 file.
        """
        self._io_handler = io_handler
        self._masks = masks
        self._h5_io_handler = h5_io_handler

    def build_wet_costs(self, bathy_tiff: str, bins: List[RangeConfig],
                        out_filename: str = WET_COSTS_TIFF,
                        wet_layer_name: str = WET_COSTS_H5_LAYER_NAME):
        """
        Build complete offshore costs. This is currently very simple. In the
        future, costs will also vary with distance to port.

        Parameters
        ----------
        bathy_tiff : path-like
            Bathymetric depth GeoTIFF. Values underwater should be negative.
        bins : list
            List of bins to use for assigning depth based costs.
        out_filename : str, optional
            Output raster with binned costs. By default, ``"wet_costs.tif"``.
        wet_layer_name : str
            Name for wet costs in H5 file
        """
        values = self._io_handler.load_tiff(bathy_tiff)
        output = self._assign_values_by_bins(values, bins)
        output[~self._masks.wet_mask] = 0
        self._io_handler.save_tiff(output, out_filename)

        if self._h5_io_handler is not None:
            out = self._h5_io_handler.load_tiff(out_filename, reproject=True)
            logger.debug('Writing wet costs to H5')
            self._h5_io_handler.write_to_h5(out, wet_layer_name)

    @staticmethod
    def _assign_values_by_bins(input: npt.NDArray,  # noqa: C901
                               bins: List[RangeConfig]) -> npt.NDArray:
        """
        Assign values based on binned raster values. Cells with values >= than
        'min' and < 'max' will be assigned 'cost'. One or both of 'min' and
        'max' can be specified. 'cost' must be specified.

        Parameters
        ----------
        input : array-like
            Input raster to assign values based upon.
        bins : list
            List of bins to use for assigning costs.

        Returns
        -------
        array-like
            Binned costs
        """
        for bin in bins:
            if bin.min > bin.max:
                raise AttributeError('Min is greater than max for bin config '
                                     f'{bin}.')
            if bin.min == float('-inf') and bin.max == float('inf'):
                msg = ('Bin covers all possible values, did you forget to set '
                       f'min or max? {bin}')
                logger.warning(msg)
                warn(msg)

        # Warn user of potential oversights in bin config. Look for gaps
        # between bin mins and maxes and overlapping bins.
        sorted_bins = sorted(bins, key=lambda x: x.min)
        last_max = float('-inf')
        for i, bin in enumerate(sorted_bins):
            if bin.min < last_max:
                last_bin = sorted_bins[i - 1] if i > 0 else '-infinity'
                msg = (f'Overlapping bins detected between bin {last_bin} '
                       f'and {bin}')
                logger.warning(msg)
                warn(msg)
            if bin.min > last_max:
                last_bin = sorted_bins[i - 1] if i > 0 else '-infinity'
                msg = f'Gap detected between bin {last_bin} and {bin}'
                logger.warning(msg)
                warn(msg)
            if i + 1 == len(sorted_bins):
                if bin.max < float('inf'):
                    msg = f'Gap detected between bin {bin} and infinity'
                    logger.warning(msg)
                    warn(msg)

            last_max = bin.max

        # Past guard clauses, perform binning
        output = np.zeros(input.shape, dtype=DEFAULT_DTYPE)

        for i, bin in enumerate(bins):
            logger.debug(f'Calculating costs for bin {i+1}/{len(bins)}: {bin}')
            mask = np.logical_and(input >= bin.min, input < bin.max)
            output = np.where(mask, bin.value, output)

        return output
