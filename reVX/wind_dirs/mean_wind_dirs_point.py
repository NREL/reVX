# -*- coding: utf-8 -*-
"""
reVX mean wind directions point sub-class
"""
import logging
import numpy as np

from reV.supply_curve.points import AggregationSupplyCurvePoint
from rex.renewable_resource import WindResource

logger = logging.getLogger(__name__)


class MeanWindDirectionsPoint(AggregationSupplyCurvePoint):
    """
    SC point class to compute mean wind directions
    """
    @property
    def h5(self):
        """
        h5 Resource handler object

        Returns
        -------
        _h5 : Resource
            Resource h5 handler object.
        """
        # pylint: disable=E0203,W0201
        if self._h5 is None:
            self._h5 = WindResource(self._h5_fpath)

        return self._h5

    def mean_wind_dirs(self, dset):
        """
        Calc the mean wind directions at every time-step

        Parameters
        ----------
        dset : str
            Wind direction dataset to aggregate

        Returns
        -------
        mean_wind_dirs : np.ndarray | float
            Mean wind direction masked by the binary exclusions
        """
        incl = self.include_mask_flat[self.bool_mask]
        gids = self._gids[self.bool_mask]

        arr_slice = (dset, slice(None), gids)

        angle = np.radians(self.h5[arr_slice], dtype=np.float32)
        sin = np.mean(np.sin(angle) * incl, axis=1)
        cos = np.mean(np.cos(angle) * incl, axis=1)

        mean_wind_dirs = np.degrees(np.arctan2(sin, cos))
        mask = mean_wind_dirs < 0
        mean_wind_dirs[mask] += 360

        return mean_wind_dirs

    @classmethod
    def run(cls, gid, excl, agg_h5, tm_dset, *wind_dir_dsets,
            excl_dict=None, inclusion_mask=None,
            resolution=64, excl_area=0.0081,
            exclusion_shape=None, close=True, gen_index=None):
        """
        Compute exclusions weight mean for the sc point from data

        Parameters
        ----------
        gid : int
            gid for supply curve point to analyze.
        excl : str | ExclusionMask
            Filepath to exclusions h5 or ExclusionMask file handler.
        agg_h5 : str | Resource
            Filepath to .h5 file to aggregate or Resource handler
        tm_dset : str
            Dataset name in the exclusions file containing the
            exclusions-to-resource mapping data.
        wind_dir_dsets : str
            Wind direction dataset(s) to aggregate, can supply multiple
            datasets
        excl_dict : dict | None
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
            None if excl input is pre-initialized.
        inclusion_mask : np.ndarray
            2D array pre-extracted inclusion mask where 1 is included and 0 is
            excluded. The shape of this will be checked against the input
            resolution.
        resolution : int
            Number of exclusion points per SC point along an axis.
            This number**2 is the total number of exclusion points per
            SC point.
        excl_area : float
            Area of an exclusion cell (square km).
        exclusion_shape : tuple
            Shape of the full exclusions extent (rows, cols). Inputing this
            will speed things up considerably.
        close : bool
            Flag to close object file handlers on exit.
        gen_index : np.ndarray
            Array of generation gids with array index equal to resource gid.
            Array value is -1 if the resource index was not used in the
            generation run.

        Returns
        -------
        out : dict
            Wind direction(s) and meta data aggregated to given supply curve
            point gid
        """
        kwargs = {"excl_dict": excl_dict,
                  "inclusion_mask": inclusion_mask,
                  "resolution": resolution,
                  "excl_area": excl_area,
                  "exclusion_shape": exclusion_shape,
                  "close": close,
                  "gen_index": gen_index}

        with cls(gid, excl, agg_h5, tm_dset, **kwargs) as point:
            out = {'meta': point.summary}
            for dset in wind_dir_dsets:
                out[dset] = point.mean_wind_dirs(dset)

        return out
