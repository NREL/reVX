# -*- coding: utf-8 -*-
"""
RED-E reV based tech potential tool
"""
import logging
from reV.supply_curve.exclusions import ExclusionMaskFromDict

logger = logging.getLogger(__name__)


class TechPotential(ExclusionMaskFromDict):
    """
    RED-E Tech Potential tool
    """
    def __init__(self, h5_path, cf_layer, power_density, layer_dict,
                 hsds=False, **kwargs):
        """
        Parameters
        ----------
        h5_path : str
            Path to .h5 file containing CF means and exclusion layers
        cf_layer : str
            Name of dataset in .h5 file containing cf means
        power_density : float
            Multiplier to convert CF means to generation means
        layers_dict : dcit
            Dictionary of LayerMask arugments {layer: {kwarg: value}}
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        kwargs : dict
            kwargs for ExclusionsMaskFromDict
        """
        layer_dict[cf_layer] = {"use_as_weights": True}
        super().__init__(h5_path, layer_dict, hsds=hsds, **kwargs)
        self._pd = power_density

    @property
    def generation(self):
        """
        Tech-potential as generation

        Returns
        -------
        gen : ndarray
        """
        gen = self[...]
        return gen

    def _generate_mask(self, *ds_slice):
        """
        Generate inclusion mask from exclusion layers

        Parameters
        ----------
        ds_slice : int | slice | list | ndarray
            What to extract from ds, each arg is for a sequential axis

        Returns
        -------
        gen : ndarray
            Tech-potential as generation
        """
        gen = super()._generate_mask(*ds_slice)
        gen *= self._pd
        return gen

    @classmethod
    def run(cls, h5_path, cf_layer, power_density, layer_dict,
            hsds=False, **kwargs):
        """
        compute tech-potential

        Parameters
        ----------
        h5_path : str
            Path to .h5 file containing CF means and exclusion layers
        cf_layer : str
            Name of dataset in .h5 file containing cf means
        power_density : float
            Multiplier to convert CF means to generation means
        layers_dict : dcit
            Dictionary of LayerMask arugments {layer: {kwarg: value}}
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        kwargs : dict
            kwargs for ExclusionsMaskFromDict

        Returns
        -------
        gen : ndarray
            Tech-potentail as generation
        """
        with cls(h5_path, cf_layer, power_density, layer_dict,
                 hsds=hsds, **kwargs) as f:
            gen = f.mask

        return gen
