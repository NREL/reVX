# -*- coding: utf-8 -*-
"""
Module to handle least cost xmission layers
"""
import logging

from reV.handlers.exclusions import ExclusionLayers
from reV.utilities.exceptions import HandlerKeyError
from rex.utilities.parse_keys import parse_keys

logger = logging.getLogger(__name__)


class XmissionCostsLayers(ExclusionLayers):
    """
    Handler for Transmission cost layers
    """
    def __init__(self, h5_file, capacity, hsds=False):
        """
        Parameters
        ----------
        h5_file : str | list | tuple
            .h5 file containing exclusion layers and techmap,
            or a list of h5 files
        capacity : int
            Capacity bin to extract costs for
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default None
        """
        self._capacity = capacity
        super().__init__(h5_file, hsds=hsds)

    def __getitem__(self, keys):
        ds, ds_slice = parse_keys(keys)

        if ds.lower().startswith('lat'):
            out = self._get_latitude(*ds_slice)
        elif ds.lower().startswith('lon'):
            out = self._get_longitude(*ds_slice)
        elif ds.lower() == 'costs':
            out = self._get_costs(*ds_slice)
        elif ds.lower() == 'mcp_costs':
            out = self._get_mcp_costs(*ds_slice)
        else:
            out = self._get_layer(ds, *ds_slice)

        return out

    def _get_costs(self, *ds_slice):
        """
        Compute the raw tie-line costs for the given slice

        Parameters
        ----------
        ds_slice : tuple of int | list | slice
            tuple describing slice of cost array to extract

        Returns
        -------
        costs : ndarray
            Array of raw tie-line costs
        """
        layers = ['base_costs', 'multipliers_{}mw'.format(self._capacity)]
        for layer_name in layers:
            if layer_name not in self.layers:
                msg = ('{} is needed to compute costs but is not in available '
                       'layers: {}. Please add it to {} and try again!'
                       .format(layer_name, self.layers, self.h5_file))
                logger.error(msg)
                raise HandlerKeyError(msg)

        costs = self[(layers[0], ) + ds_slice] * self[(layers[1], ) + ds_slice]

        return costs

    def _get_mcp_costs(self, *ds_slice):
        """
        Compute the MinimumCostPath costs for the given slice

        Parameters
        ----------
        ds_slice : tuple of int | list | slice
            tuple describing slice of cost array to extract

        Returns
        -------
        mcp_costs : ndarray
            Array of MinimumCostPath costs
        """
        layers = ['base_costs', 'multipliers_{}mw'.format(self._capacity),
                  'transmission_barrier']
        for layer_name in layers:
            if layer_name not in self.layers:
                msg = ('{} is needed to compute mcp_costs but is not in '
                       'available layers: {}. Please add it to {} and try '
                       'again!'
                       .format(layer_name, self.layers, self.h5_file))
                logger.error(msg)
                raise HandlerKeyError(msg)

        mcp_costs = (self[(layers[0], ) + ds_slice]
                     * self[(layers[1], ) + ds_slice]
                     + self[(layers[2], ) + ds_slice])

        return mcp_costs
