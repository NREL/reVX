# -*- coding: utf-8 -*-
"""
Resource Extraction Tools
"""
import gzip
import logging
import numpy as np
import os
import pandas as pd
import pickle
from reV.handlers.resource import NSRDB, Resource, SolarResource, WindResource
from scipy.spatial import cKDTree

BIN = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
BIN = os.path.join(os.path.dirname(BIN), 'bin')
logger = logging.getLogger(__name__)


class ResourceX(Resource):
    """
    Resource data extraction tool
    """
    def __init__(self, res_h5, tree=None, **kwargs):
        """
        Parameters
        ----------
        res_h5 : str
            Path to resource .h5 file of interest
        tree : str
            path to .pgz file containing pickled cKDTree of lat, lon
            coordinates
        kwargs : dict
            Kwargs for Resource
        """
        super().__init__(res_h5, **kwargs)
        self._tree = tree

    @property
    def tree(self):
        """
        Returns
        -------
        tree : cKDTree
            Lat, lon coordinates cKDTree
        """
        if not isinstance(self._tree, cKDTree):
            self._tree = self._init_tree(tree=self._tree)

        return self._tree

    @property
    def countries(self):
        """
        Returns
        -------
        countries : ndarray
            Countries available in .h5 file
        """
        if 'country' in self.meta:
            countries = self.meta['country'].unique()
        else:
            countries = None

        return countries

    @property
    def states(self):
        """
        Returns
        -------
        states : ndarray
            States available in .h5 file
        """
        if 'state' in self.meta:
            states = self.meta['state'].unique()
        else:
            states = None

        return states

    @property
    def counties(self):
        """
        Returns
        -------
        counties : ndarray
            Counties available in .h5 file
        """
        if 'county' in self.meta:
            counties = self.meta['county'].unique()
        else:
            counties = None

        return counties

    @staticmethod
    def _load_tree(tree_pickle):
        """
        Load tree from pickle file

        Parameters
        ----------
        tree_pickle : str
            Pickle (.pkl, .pickle) or compressed pickle (.pgz, .pgzip) file
            containing precomputed cKDTree

        Returns
        -------
        tree : cKDTree
            Precomputed tree of lat, lon coordinates
        """
        try:
            if tree_pickle.endswith(('.pkl', '.pickle')):
                with open(tree_pickle, 'rb') as f:
                    tree = pickle.load(f)
            elif tree_pickle.endswith(('.pgz', '.pgzip', '.gz', '.gzip')):
                with gzip.open(tree_pickle, 'r') as f:
                    tree = pickle.load(f)
            else:
                logger.warning('Cannot parse files of type "{}"'
                               .format(tree_pickle))
                tree = None
        except Exception as e:
            logger.warning('Could not extract tree from {}: {}'
                           .format(tree_pickle, e))
            tree = None

        return tree

    def _init_tree(self, tree=None):
        """
        Inititialize cKDTree of lat, lon coordinates

        Parameters
        ----------
        tree : str | cKDTree | NoneType
            Path to .pgz file containing pre-computed tree
            If None search bin for .pgz file matching h5 file
            else compute tree

        Returns
        -------
        tree : cKDTree
            cKDTree of lat, lon coordinate from wtk .h5 file
        """
        if not isinstance(tree, (cKDTree, str, type(None))):
            tree = None
            logger.warning('Precomputed tree must be supplied as a pickle '
                           'file or a cKDTree, not a {}'
                           .format(type(tree)))

        if tree is None:
            pgz_files = [file for file in os.listdir(BIN)
                         if file.endswith('.pgz')]
            for pgz in pgz_files:
                prefix = pgz.split('_tree')[0]
                if self._h5_file.startswith(prefix):
                    tree = os.path.join(BIN, pgz)
                    break

        if isinstance(tree, str):
            tree = self._load_tree(tree)

        if tree is None:
            lat_lon = self.meta[['latitude', 'longitude']].values
            tree = cKDTree(lat_lon)

        return tree

    def _get_nearest(self, lat_lon):
        """
        Get nearest gid to given (lat, lon) pair or pairs

        Parameters
        ----------
        lat_lon : ndarray
            Either a single (lat, lon) pair or series of (lat, lon) pairs

        Returns
        -------
        gids : int | ndarray
            Nearest gid(s) to given (lat, lon) pair(s)
        """
        _, gids = self.tree.query(lat_lon)
        return gids

    def _get_region(self, region, region_col='state'):
        """
        Get the gids for given region

        Parameters
        ----------
        region : str
            Region to search for
        region_col : str
            Region column to search

        Returns
        -------
        gids : ndarray
            Vector of gids in given region
        """
        gids = (self.meta[region_col] == region).index.values
        return gids

    def _get_timestep_idx(self, timestep):
        """
        Get the index of the desired timestep

        Parameters
        ----------
        timestep : str
            Timestep of interest

        Returns
        -------
        ts_idx : int
            Time index value
        """
        timestep = pd.to_datetime(timestep)
        idx = np.where(self.time_index == timestep)[0][0]

        return idx

    def get_site_ts(self, ds_name, lat_lon):
        """
        Extract timeseries of nearest site(s) to given lat_lon

        Parameters
        ----------
        ds_name : str
            Dataset to extract
        lat_lon : tuple | list
            (lat, lon) coordinate of interest or pairs of coordinates

        Return
        ------
        site_ts : ndarray
            Time-series for given site(s) and dataset
        """
        gid = self._get_nearest(lat_lon)
        site_ts = self[ds_name, :, gid]

        return site_ts

    def get_site_df(self, ds_name, lat_lon):
        """
        Extract timeseries of nearest site to given(s) lat_lon and return as
        a DataFrame

        Parameters
        ----------
        ds_name : str
            Dataset to extract
        lat_lon : tuple
            (lat, lon) coordinate of interest

        Return
        ------
        site_df : pandas.DataFrame
            Time-series DataFrame for given site and dataset
        """
        gid = self._get_nearest(lat_lon)
        if isinstance(gid, int):
            site_df = pd.DataFrame(index=self.time_index)
            site_df.name = gid
            site_df.index.name = 'time_index'
            site_df[ds_name] = self[ds_name, :, gid]
        else:
            site_df = pd.DataFrame(columns=gid, index=self.time_index)
            site_df.name = ds_name
            site_df.index.name = 'time_index'
            site_df.loc[:, :] = self[ds_name, :, gid]

        return site_df

    def get_region_ts(self, ds_name, region, region_col='state'):
        """
        Extract timeseries of of all sites in given region

        Parameters
        ----------
        ds_name : str
            Dataset to extract
        region : str
            Region to search for
        region_col : str
            Region column to search

        Return
        ------
        region_ts : ndarray
            Time-series array of desired dataset for all sites in desired
            region
        """
        gids = self._get_region(region, region_col=region_col)
        region_ts = self[ds_name, :, gids]

        return region_ts

    def get_region_df(self, ds_name, region, region_col='state'):
        """
        Extract timeseries of of all sites in given region and return as a
        DataFrame

        Parameters
        ----------
        ds_name : str
            Dataset to extract
        region : str
            Region to extract all pixels for
        region_col : str
            Region column to search

        Return
        ------
        region_df : pandas.DataFrame
            Time-series array of desired dataset for all sites in desired
            region
        """
        gids = self._get_region(region, region_col=region_col)
        region_df = pd.DataFrame(columns=gids, index=self.time_index)
        region_df.name = ds_name
        region_df.index.name = 'time_index'
        region_df.loc[:, :] = self[ds_name, :, gids]

        return region_df

    def get_SAM_df(self, lat_lon):
        """
        Extract time-series of all variables needed to run SAM for nearest
        site to given lat_lon

        Parameters
        ----------
        lat_lon : tuple
            (lat, lon) coordinate of interest

        Return
        ------
        SAM_df : pandas.DataFrame | list
            Time-series DataFrame for given site and dataset
            If multiple lat, lon pairs are given a list of DatFrames is
            returned
        """
        gids = self._get_nearest(lat_lon)
        if isinstance(gids, int):
            gids = [gids, ]

        SAM_df = []
        for gid in gids:
            SAM_df.append(self['SAM', gid])

        if len(SAM_df) == 1:
            SAM_df = SAM_df[0]

        return SAM_df

    def get_timestep_map(self, ds_name, timestep, region=None,
                         region_col='state'):
        """
        Extract a map of the given dataset at the given timestep for the
        given region if supplied

        Parameters
        ----------
        ds_name : str
            Dataset to extract
        timestep : str
            Timestep of interest
        region : str
            Region to extract all pixels for
        region_col : str
            Region column to search

        Returns
        -------
        ts_map : pandas.DataFrame
            DataFrame of map values
        """
        lat_lons = self.meta[['latitude', 'longitude']].values
        ts_idx = self._get_timestep_idx(timestep)
        gids = slice(None)
        if region is not None:
            gids = self._get_region(region, region_col=region_col)
            lat_lons = lat_lons[gids]

        ts_map = self[ds_name, ts_idx, gids]
        ts_map = pd.DataFrame({'longitude': lat_lons[:, 1],
                               'latitude': lat_lons[:, 0],
                               ds_name: ts_map})

        return ts_map


class SolarX(SolarResource, ResourceX):
    """
    Solar Resource extraction class
    """
    def __init__(self, solar_h5, tree=None, **kwargs):
        """
        Parameters
        ----------
        solar_h5 : str
            Path to solar .h5 file of interest
        tree : str
            path to .pgz file containing pickled cKDTree of lat, lon
            coordinates
        kwargs : dict
            Kwargs for Resource
        """
        super().__init__(solar_h5, **kwargs)
        self._tree = self._init_tree(tree=tree)


class NSRDBX(NSRDB, ResourceX):
    """
    NSRDB extraction class
    """
    def __init__(self, nsrdb_h5, tree=None, **kwargs):
        """
        Parameters
        ----------
        nsrdb_h5 : str
            Path to NSRDB .h5 file of interest
        tree : str
            path to .pgz file containing pickled cKDTree of lat, lon
            coordinates
        kwargs : dict
            Kwargs for Resource
        """
        super().__init__(nsrdb_h5, **kwargs)
        self._tree = self._init_tree(tree=tree)


class WindX(WindResource, ResourceX):
    """
    Wind Resource extraction class
    """
    def __init__(self, wind_h5, tree=None, **kwargs):
        """
        Parameters
        ----------
        wind_h5 : str
            Path to Wind .h5 file of interest
        tree : str
            path to .pgz file containing pickled cKDTree of lat, lon
            coordinates
        kwargs : dict
            Kwargs for Resource
        """
        super().__init__(wind_h5, **kwargs)
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
