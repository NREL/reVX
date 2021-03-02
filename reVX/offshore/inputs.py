# -*- coding: utf-8 -*-
"""
Extract offshore inputs from exclusion layers
"""
import logging
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass

from rex.resource import Resource
from rex.utilities.utilities import parse_table
from reV.handlers.exclusions import ExclusionLayers

logger = logging.getLogger(__name__)


class OffshoreInputs:
    """
    Class to extract offshore inputs from exclusion .h5
    """
    def __init__(self, excl_h5, offshore_sites, tm_dset='techmap_wtk'):
        self._excl_h5 = excl_h5
        self._offshore_meta = self._create_offshore_meta(offshore_sites,
                                                         tm_dset)

    @staticmethod
    def _reduce_tech_map(excl_h5, tm_dset='techmap_wtk', offshore_gids=None):
        """
        Find the row and column indices that correspond to the centriod of
        each offshore gid in exclusions layers. If offshore gids are not
        provided the centroid of every gid is in techmap.

        Parameters
        ----------
        excl_h5 : str
            Path to
        tm_dset : str, optional
            [description], by default 'techmap_wtk'
        offshore_gids : [type], optional
            [description], by default None

        Returns
        -------
        tech_map : pandas.DataFrame
            DataFrame mapping resource gid to exclusions latitude, longitude, r
        """
        with ExclusionLayers(excl_h5) as f:
            tech_map = f[tm_dset]

        if offshore_gids is None:
            offshore_gids = np.unique(tech_map)

        tech_map = np.array(center_of_mass(tech_map, labels=tech_map,
                                           index=offshore_gids),
                            dtype='float32')

        index = pd.Index(offshore_gids, name='res_gid')
        tech_map = pd.DataFrame(tech_map, columns=['row_id', 'col_id'],
                                index=index)
        tech_map['latitudes'] = f['latitude',
                                  tech_map['row_id'].values,
                                  tech_map['col_id'].values]
        tech_map['longitude'] = f['longitude',
                                  tech_map['row_id'].values,
                                  tech_map['col_id'].values]

        return tech_map

    @staticmethod
    def _parse_offshore_sites(offshore_sites):
        """
        Load offshore sites from disc if needed

        Parameters
        ----------
        offshore_sites : str | pandas.DataFrame
            Path to .csv file with offshore sites or offshore meta, or path
            to a .h5 file to extact site meta from, or pre-extracted site meta
            DataFrame

        Returns
        -------
        offshore_sites : pandas.DataFrame
            Offshore site meta data
        """
        if isinstance(offshore_sites, str):
            if offshore_sites.endswith('.h5'):
                with Resource(offshore_sites) as f:
                    offshore_sites = f.meta
            else:
                offshore_sites = parse_table(offshore_sites)

        if not isinstance(offshore_sites, pd.DataFrame):
            msg = ("offshore sites must be a .csv, .json, or .h5 file path, "
                   "or a pre-extracted pandas DataFrame, but {} was provided"
                   .format(offshore_sites))
            logger.error(msg)
            raise ValueError(msg)

        if 'offshore' in offshore_sites:
            mask = offshore_sites['offshore'] == 1
            offshore_sites = offshore_sites.loc[mask]

        return offshore_sites

    def _create_offshore_meta(self, offshore_sites, tm_dset='techmap_wtk'):
        """
        [summary]

        Parameters
        ----------
        offshore_sites : [type]
            [description]
        tm_dset : str, optional
            [description], by default 'techmap_wtk'

        Returns
        -------

        """
        offshore_sites = self._parse_offshore_sites(offshore_sites)
        if 'gid' in offshore_sites:
            offshore_gids = offshore_sites['gid'].values
        else:
            offshore_gids = None

        tech_map = self._reduce_tech_map(self._excl_h5, tm_dset=tm_dset,
                                         offshore_gids=offshore_gids)

        return tech_map
