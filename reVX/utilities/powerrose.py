# -*- coding: utf-8 -*-
"""
Aggregate powerrose and sort directions by dominance
"""
from reV.supply_curve.aggregation import Aggregation


class PowerRoseDirections(Aggregation):
    """
    Aggregate PowerRose to Supply Curve points and sort directions in order
    of prominence using using following key:
    [[0, 1, 2]
     [7, x, 3],
     [6, 5, 4]]
    """

    def __init__(self, power_rose_h5_fpath, excl_fpath,
                 agg_dset='powerrose_100m', tm_dset='',
                 resolution=64):
        """
        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        h5_fpath : str
            Filepath to .h5 file to aggregate
        tm_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        agg_dset : str
            Dataset to aggreate, can supply multiple datasets
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        """
        super().__init__(excl_fpath, power_rose_h5_fpath, tm_dset, agg_dset,
                         resolution=resolution)
