# -*- coding: utf-8 -*-
"""
Extract offshore inputs from exclusion layers
"""
import logging
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
from scipy.spatial import cKDTree

from rex.resource import Resource
from rex.utilities.utilities import parse_table, get_lat_lon_cols
from reV.handlers.exclusions import ExclusionLayers

logger = logging.getLogger(__name__)


class OffshoreInputs:
    """
    Class to extract offshore inputs from exclusion .h5
    """
    def __init__(self, excl_fpath, offshore_sites, tm_dset='techmap_wtk'):
        self._excl_fpath = excl_fpath
        self._offshore_meta = self._create_offshore_meta(offshore_sites,
                                                         tm_dset)

    @staticmethod
    def _reduce_tech_map(excl_fpath, tm_dset='techmap_wtk',
                         offshore_gids=None):
        """
        Find the row and column indices that correspond to the centriod of
        each offshore gid in exclusions layers. If offshore gids are not
        provided the centroid of every gid is in techmap.

        Parameters
        ----------
        excl_fpath : str
            Path to exclusions .h5 file
        tm_dset : str, optional
            Dataset / layer name for wind toolkit techmap,
            by default 'techmap_wtk'
        offshore_gids : ndarray | list, optional
            Vector or list of offshore gids, by default None

        Returns
        -------
        tech_map : pandas.DataFrame
            DataFrame mapping resource gid to exclusions latitude, longitude,
            row index, column index
        """
        with ExclusionLayers(excl_fpath) as f:
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
        Create offshore meta from offshore sites and techmap

        Parameters
        ----------
        offshore_sites : str | pandas.DataFrame
            Path to .csv file with offshore sites or offshore meta, or path
            to a .h5 file to extact site meta from, or pre-extracted site meta
            DataFrame
        tm_dset : str, optional
            Dataset / layer name for wind toolkit techmap,
            by default 'techmap_wtk'

        Returns
        -------
        offshore_sites : pandas.DataFrames
            Offshore site meta data including mapping to exclusion
        """
        offshore_sites = self._parse_offshore_sites(offshore_sites)
        if 'gid' in offshore_sites:
            offshore_gids = offshore_sites['gid'].values
            offshore_sites = offshore_sites.set_index('gid')
        elif 'res_gid' in offshore_sites:
            offshore_gids = offshore_sites['res_gid'].values
            offshore_sites = offshore_sites.set_index('res_gid')
        else:
            offshore_gids = None

        tech_map = self._reduce_tech_map(self._excl_fpath, tm_dset=tm_dset,
                                         offshore_gids=offshore_gids)

        if offshore_gids is not None:
            offshore_meta = offshore_sites.join(tech_map)
        else:
            cols = ['latitude', 'longitude']
            # pylint: disable=not-callable
            tree = cKDTree(tech_map[cols].values)
            cols = get_lat_lon_cols(offshore_sites)
            _, pos = tree.query(offshore_sites[cols])

            cols = ['res_gid', 'row_id', 'col_id']
            tech_map = tech_map.reset_index().iloc[pos][cols]
            offshore_meta = pd.concat([offshore_sites, tech_map])

        return offshore_meta
