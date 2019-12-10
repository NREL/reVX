# -*- coding: utf-8 -*-
"""
RED-E reV based tech potential tool
"""
import logging
from reV.supply_curve.exclusions import ExclusionMask, LayerMask

logger = logging.getLogger(__name__)


class TechPotential:
    """
    RED-E Tech Potential tool
    """
    def __init__(self, h5_path, cf_layer, power_density):
        """
        Parameters
        ----------
        h5_path : str
            Path to .h5 file containing CF means and exclusion layers
        cf_layer : str
            Name of dataset in .h5 file containing cf means
        power_density : float
            Multiplier to convert CF means to generation means
        """
        cf_layer = LayerMask(cf_layer, use_as_weights=True)
        self._h5 = ExclusionMask(h5_path, cf_layer)
        self._gen = self._h5.mask * power_density
        del self._h5._layers[cf_layer.layer]

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._h5.close()

        if type is not None:
            raise

    def __repr__(self):
        msg = ("{} from {}"
               .format(self.__class__.__name__, self._h5))
        return msg
