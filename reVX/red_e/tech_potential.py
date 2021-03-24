# -*- coding: utf-8 -*-
"""
RED-E reV based tech potential tool
"""
import logging

from reV.supply_curve.exclusions import ExclusionMaskFromDict
from reVX.utilities.utilities import log_versions

logger = logging.getLogger(__name__)


class TechPotential(ExclusionMaskFromDict):
    """
    RED-E Tech Potential tool
    """
    def __init__(self, h5_path, base_layer, excl_dict, power_density=1,
                 hsds=False, min_area=None, kernel='queen',
                 check_layers=False):
        """
        Parameters
        ----------
        h5_path : str
            Path to .h5 file containing CF means and exclusion layers
        base_layer : str
            Name of dataset in .h5 file containing base layer
        excl_dict : dcit
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
        power_density : float
            Multiplier to convert CF means to generation means
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        min_area : float | NoneType
            Minimum required contiguous area in sq-km
        kernel : str
            Contiguous filter method to use on final exclusion
        check_layers : bool
            Run a pre-flight check on each layer to ensure they contain
            un-excluded values
        """
        log_versions(logger)
        layers_dict = excl_dict.copy()
        layers_dict[base_layer] = {"use_as_weights": True}
        super().__init__(h5_path, layers_dict=layers_dict, hsds=hsds,
                         min_area=min_area, kernel=kernel,
                         check_layers=check_layers)
        self._pd = power_density

    @property
    def profile(self):
        """
        GeoTiff profile for exclusions

        Returns
        -------
        profile : dict
            Generic GeoTiff profile for exclusions in .h5 file
        """
        return self.excl_h5.profile

    @property
    def generation(self):
        """
        Tech-potential as generation

        Returns
        -------
        gen : ndarray
        """
        gen = self[...]

        return gen * self._pd

    @staticmethod
    def _parse_sub_slice(sub_slice):
        """
        Convert sub_slice tuples into a tuple of slice objects

        Parameters
        ----------
        sub_slice : tuple | None
            Tuple of sub-window coordinates in the format (start, end, step),
            if None return (slice(None), )

        Returns
        -------
        out_slice : tuple
            Tuple of equivalent slice objects
        """
        if sub_slice is not None:
            if not isinstance(sub_slice[0], (list, tuple)):
                sub_slice = (sub_slice, )

            out_slice = ()
            for s in sub_slice:
                out_slice += (slice(*s), )
        else:
            out_slice = (slice(None), )

        return out_slice

    @classmethod
    def run(cls, h5_path, base_layer, excl_dict, sub_slice=None,
            power_density=1, hsds=False, min_area=None, kernel='queen',
            check_layers=False, generation=False):
        """
        compute tech-potential

        Parameters
        ----------
        h5_path : str
            Path to .h5 file containing CF means and exclusion layers
        base_layer : str
            Name of dataset in .h5 file containing base layer
        excl_dict : dcit
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
        sub_slice : tuple, optional
            Coordinates of sub-slice to extract, if None extract entire mask,
            by default None
        power_density : float, optional
            Multiplier to convert CF means to generation means, by default 1
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        min_area : float | NoneType, optional
            Minimum required contiguous area in sq-km, by default None
        kernel : str, optional
            Contiguous filter method to use on final exclusion,
            by default 'queen'
        check_layers : bool, optional
            Run a pre-flight check on each layer to ensure they contain
            un-excluded values, by default False
        generation : bool, optional
            Flag to return generation instead of base layer:
            gen = base layer * power density
            by default False

        Returns
        -------
        out : ndarray
            Base layer with exclusions masked out, if generation is true
            multiply base layer mask by power density
        """
        with cls(h5_path, base_layer, excl_dict, power_density=power_density,
                 hsds=hsds, min_area=min_area, kernel=kernel,
                 check_layers=check_layers) as f:
            sub_slice = cls._parse_sub_slice(sub_slice)
            out = f[sub_slice]

            if generation:
                out *= f._pd

        return out
