"""
Create wet (offshore) costs and save to GeoTIFF.
"""
import logging
from typing import List
from typing_extensions import TypedDict, Required

import numpy as np
import numpy.typing as npt
from reVX.least_cost_xmission.config.constants import DEFAULT_DTYPE, \
    WET_COSTS_TIFF

from reVX.least_cost_xmission.transmission_layer_io_handler import \
    TransLayerIoHandler

logger = logging.getLogger(__name__)


class BinConfig(TypedDict, total=False):
    """
    Config for assigning cost based on bins. Cells with values >= than 'min'
    and < 'max' will be assigned 'cost'. One or both of 'min' and 'max' can be
    specified.
    """
    min: float
    max: float
    cost: Required[float]


class WetCostCreator:
    """
    Create offshore costs and save to GeoTIFF.
    """
    def __init__(self, io_handler: TransLayerIoHandler):
        """
        Parameters
        ----------
        io_handler
            Transmission layer IO handler
        """
        self._io_handler = io_handler

    def build_wet_costs(self, bathy_tiff: str, bins: List[BinConfig],
                        out_filename: str = WET_COSTS_TIFF):
        """
        Build complete offshore costs. This is currently very simple. In the
        future, costs will also vary with distance to port.

        Parameters
        ----------
        bathy_tiff
            Bathymetric depth GeoTIFF. Values underwater should be negative.
        bins
            List of bins to use for assigning depth based costs.
        out_filename
            Output raster with binned costs.
        """
        self.assign_cost_by_bins(bathy_tiff, bins, out_filename)

    def assign_cost_by_bins(self, in_filename: str, bins: List[BinConfig],
                            out_filename: str):
        """
        Assign costs based on binned raster values. Cells with values >= than
        'min' and < 'max' will be assigned 'cost'. One or both of 'min' and
        'max' can be specified. 'cost' must be specified.

        Parameters
        ----------
        in_filename
            Input raster to assign costs based upon.
        bins
            List of bins to use for assigning costs.
        out_filename
            Output raster with binned costs.
        """
        input = self._io_handler.load_tiff(in_filename)

        output = self._assign_values_by_bins(input, bins)
        self._io_handler.save_tiff(output, out_filename)

    @staticmethod
    def _assign_values_by_bins(input: npt.NDArray,  # noqa: C901
                               bins: List[BinConfig]) -> npt.NDArray:
        """
        Assign values based on binned raster values. Cells with values >= than
        'min' and < 'max' will be assigned 'cost'. One or both of 'min' and
        'max' can be specified. 'cost' must be specified.

        Parameters
        ----------
        input
            Input raster to assign values based upon.
        bins
            List of bins to use for assigning costs.

        Returns
        -------
            Binned costs
        """
        for bin in bins:
            if ('min' not in bin) and ('max' not in bin):
                raise AttributeError(f'Bin config {bin} requires "min", "max",'
                                     ' or both.')
            if ('min' in bin) and ('max' in bin) and (bin['min'] > bin['max']):
                raise AttributeError('Min is greater than max for bin config '
                                     f'{bin}.')

        # Warn user of potential oversights in bin config. Look for gaps
        # between bin mins and maxes and overlapping bins.
        sorted_bins = sorted(bins, key=lambda x: x.get('min', float('-inf')))
        last_max = float('-inf')
        for i, bin in enumerate(sorted_bins):
            if bin.get('min', float('-inf')) < last_max:
                last_bin = sorted_bins[i - 1] if i > 0 else '-infinity'
                msg = (f'Overlapping bins detected between bin {last_bin} '
                       f'and {bin}')
                logger.warning(msg)
            if bin.get('min', float('-inf')) > last_max:
                last_bin = sorted_bins[i - 1] if i > 0 else '-infinity'
                msg = f'Gap detected between bin {last_bin} and {bin}'
                logger.warning(msg)

            if i + 1 == len(sorted_bins):
                if bin.get('max', float('inf')) < float('inf'):
                    msg = f'Gap detected between bin {bin} and infinity'
                    logger.warning(msg)

            last_max = bin.get('max', float('inf'))

        # Past guard clauses, perform binning
        output = np.zeros(input.shape, dtype=DEFAULT_DTYPE)

        for i, bin in enumerate(bins):
            logger.debug(f'Calculating costs for bin {i+1}/{len(bins)}: {bin}')
            if ('min' in bin) and ('max' not in bin):
                output = np.where(input >= bin['min'], bin['cost'], output)
            elif ('min' not in bin) and ('max' in bin):
                output = np.where(input < bin['max'], bin['cost'], output)
            elif ('min' in bin) and ('max' in bin):
                mask = np.logical_and(input >= bin['min'], input < bin['max'])
                output = np.where(mask, bin['cost'], output)

        return output
