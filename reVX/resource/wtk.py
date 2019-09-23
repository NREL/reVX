# -*- coding: utf-8 -*-
"""
WIND Toolkit (WTK) Extraction tool
"""
import logging
from reV.handlers.resource import WindResource

from reVX.resource.base import ResourceX

logger = logging.getLogger(__name__)


class WTKX(WindResource, ResourceX):
    """
    WIND Toolkit extraction class
    """
    def __init__(self, wtk_h5, tree=None, **kwargs):
        """
        Parameters
        ----------
        wtk_h5 : str
            Path to WIND Toolkit .h5 file of interest
        tree : str
            path to .pgz file containing pickled cKDTree of lat, lon
            coordinates
        kwargs : dict
            Kwargs for Resource
        """
        super().__init__(wtk_h5, **kwargs)
        self._tree = self._init_tree(tree)

    def get_SAM_df(self, hub_height, lat_lon, **kwargs):
        """
        Extract time-series of all variables needed to run SAM for nearest
        site to given lat_lon and hub height

        Parameters
        ----------
        hub_height : int
            Hub height of interest
        lat_lon : tuple
            (lat, lon) coordinate of interest
        kwargs : dict
            Internal kwargs for _get_SAM_df:
            - require_wind_dir
            - icing

        Return
        ------
        SAM_df : pandas.DataFrame | list
            Time-series DataFrame for given site and dataset
            If multiple lat, lon pairs are given a list of DatFrames is
            returned
        """
        ds_name = 'SAM_{}m'.format(hub_height)
        gids = self._get_nearest(lat_lon)
        if isinstance(gids, int):
            gids = [gids, ]

        SAM_df = []
        for gid in gids:
            SAM_df.append(self._get_SAM_df(ds_name, gid, **kwargs))

        if len(SAM_df) == 1:
            SAM_df = SAM_df[0]

        return SAM_df
