"""
Create wet (offshore) costs and save to GeoTIFF.
"""
import logging
from typing import List
from warnings import warn

import numpy as np
import numpy.typing as npt
from reVX.least_cost_xmission.layers.base import LayerCreator
from reVX.config.transmission_layer_creation import RangeConfig
from reVX.least_cost_xmission.config.constants import (WET_COSTS_TIFF,
                                                       WET_COSTS_H5_LAYER_NAME)

logger = logging.getLogger(__name__)


class WetCostCreator(LayerCreator):
    """
    Create offshore costs and save to GeoTIFF.
    """

    def build(self, bathy_tiff: str, bins: List[RangeConfig],
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
        wet_layer_name : str
            Name for wet costs in H5 file
        """
        values = self._io_handler.load_data_using_h5_profile(
            bathy_tiff, reproject=True)
        output = self._assign_values_by_bins(values, bins)
        output[~self._mask] = 0

        out_filename = self.output_tiff_dir / WET_COSTS_TIFF
        self._io_handler.save_data_using_h5_profile(output, out_filename)

        if self._io_handler is not None:
            out = self._io_handler.load_data_using_h5_profile(
                out_filename, reproject=True)
            logger.debug('Writing wet costs to H5')
            self._io_handler.write_layer_to_h5(out, wet_layer_name)

    def _assign_values_by_bins(self, input: npt.NDArray,  # noqa: C901
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
        output = np.zeros(input.shape, dtype=self._dtype)

        for i, bin in enumerate(bins):
            logger.debug(f'Calculating costs for bin {i+1}/{len(bins)}: {bin}')
            mask = np.logical_and(input >= bin.min, input < bin.max)
            output = np.where(mask, bin.value, output)

        return output
