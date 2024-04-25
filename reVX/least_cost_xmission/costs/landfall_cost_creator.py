"""
Create wet (offshore) costs and save to GeoTIFF.
"""
import logging

import numpy as np

from reVX.least_cost_xmission.layers.base import LayerCreator
from reVX.least_cost_xmission.config.constants import (
    LANDFALL_COSTS_TIFF, LANDFALL_COSTS_H5_LAYER_NAME
)

logger = logging.getLogger(__name__)


class LandfallCostCreator(LayerCreator):
    """
    Create landfall costs and save to GeoTIFF.
    """

    def build(self, landfall_cost: float,
              landfall_layer_name: str = LANDFALL_COSTS_H5_LAYER_NAME):
        """Build landfall costs.

        Currently, this just sets the landfall mask to the given
        landfall cost value (0's everywhere else).

        Parameters
        ----------
        bathy_tiff : path-like
            Bathymetric depth GeoTIFF. Values underwater should be negative.
        bins : list
            List of bins to use for assigning depth based costs.
        wet_layer_name : str
            Name for wet costs in H5 file
        """
        landfall_costs = np.zeros(self._io_handler.shape, dtype=self._dtype)
        landfall_costs[self._mask] = landfall_cost

        out_filename = self.output_tiff_dir / LANDFALL_COSTS_TIFF
        self._io_handler.save_data_using_h5_profile(landfall_costs,
                                                    out_filename)

        if self._io_handler is not None:
            logger.debug('Writing landfall costs to H5')
            self._io_handler.write_layer_to_h5(landfall_costs,
                                               landfall_layer_name)
