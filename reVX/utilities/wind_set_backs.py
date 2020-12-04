# -*- coding: utf-8 -*-
"""
Handler to convert exclusion to/from .h5 and .geotiff
"""
from abc import ABC
import geopandas as gpd
import h5py
import json
import logging
import numpy as np
from rasterio import features
from shapely.geometry import shape
from warnings import warn

from rex.utilities import parse_table
from reV.handlers.exclusions import ExclusionLayers

logger = logging.getLogger(__name__)


class BaseWindSetBacks(ABC):
    """
    Create exclusions layers for wind set-backs
    """
    MULTIPLIERS = {'high': 3, 'moderate': 1.1}

    def __init__(self, excl_h5, hub_height, rotor_diameter, regs_fpath=None,
                 multiplier=None, hsds=False, chunks=(128, 128)):
        """
        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing or to contain exclusion layers
        hub_height : float | int
            Turbine hub height (m), used along with rotor diameter to compute
            blade tip height which is used to determine set-back distance
        rotor_diameter : float | int
            Turbine rotor diameter (m), used along with hub height to compute
            blade tip height which is used to determine set-back distance
        regs_fpath : str | None, optional
            Path to wind regulations .csv file, if None create global
            set-backs, by default None
        multiplier : int | float | str | None, optional
            Set-back multiplier to use if wind regulations are not supplied,
            if str, must one of {'high': 3, 'moderate': 1.1},
            by default None
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        chunks : tuple, optional
            Chunk size to use for set-back layers, if None use default chunk
            size in excl_h5, by default (128, 128)
        """
        self._excl_h5 = excl_h5
        self._hub_height = hub_height
        self._rotor_diameter = rotor_diameter
        self._hsds = hsds
        self._shape, self._chunks, self._profile = \
            self._parse_excl_properties(excl_h5, chunks, hsds=hsds)

        self._regs, self._multi = self._preflight_check(regs_fpath, multiplier)

    def __repr__(self):
        msg = "{} for {}".format(self.__class__.__name__, self._excl_h5)
        return msg

    def __setitem__(self, layer, arr):
        """
        Write layer to excl_h5 file

        Parameters
        ----------
        layer : str
            Layer to set
        arr : ndarray
            Path to GeoTiff to load data from
        """

        if layer in self.layers:
            msg = "{} is already present in {}".format(layer, self._excl_h5)
            logger.error(msg)
            raise KeyError(msg)

        self._write_layer(layer, arr, chunks=self._chunks)

    @property
    def hub_height(self):
        """
        Turbine hub-height in meters

        Returns
        -------
        float
        """
        return self._hub_height

    @property
    def rotor_diameter(self):
        """
        Turbine rotor diameter in meters

        Returns
        -------
        float
        """
        return self._rotor_diameter

    @property
    def tip_height(self):
        """
        Turbine blade tip height in meters

        Returns
        -------
        float
        """
        return self._hub_height + self._rotor_diameter / 2

    @property
    def layers(self):
        """
        Available exclusion layers in .h5 file

        Returns
        -------
        layers : list
            Available layers in .h5 file
        """
        with ExclusionLayers(self._excl_h5, hsds=self._hsds) as exc:
            layers = exc.layers

        return layers

    @staticmethod
    def _parse_excl_properties(excl_h5, chunks, hsds=False):
        """
        Parse exclusions shape, chunk size, and profile from excl_h5 file

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing or to contain exclusion layers
        chunks : tuple | None
            Chunk size of exclusions datasets
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False

        Returns
        -------
        shape : tuple
            Shape of exclusions datasets
        chunks : tuple | None
            Chunk size of exclusions datasets
        profile : str
            GeoTiff profile for exclusions datasets
        """
        with ExclusionLayers(excl_h5, hsds=hsds) as exc:
            shape = exc.shape
            profile = exc.profile
            if chunks is None:
                chunks = exc.chunks

        if len(chunks) < 3:
            chunks = (1, ) + chunks

        if len(shape) < 3:
            shape = (1, ) + shape

        return shape, chunks, profile

    @staticmethod
    def _parse_regs(regs_fpath, excl_h5):
        regs = parse_table(regs_fpath)
        if 'FIPS' not in regs:
            msg = ('Wind regulations does not have county FIPS! Please add a '
                   '"FIPS" columns with the unique county FIPS values.')
            logger.error(msg)
            raise RuntimeError(msg)

        with ExclusionLayers(excl_h5) as exc:
            fips = exc['cnty_fips']
            transform = exc.profile['transform']
            crs = exc.crs

        fips_df = gpd.GeoDataFrame(columns=['geometry', 'FIPS'], crs=crs)
        for i, (p, v) in enumerate(features.shapes(fips, transform=transform)):
            fips_df.at[i] = shape(p), v

        fips_check = regs['FIPS'].isin(fips_df['FIPS'])
        if not fips_check.all():
            msg = ('The following county FIPS were requested in by the wind '
                   'regulations but were not availble in the Exclusions '
                   '"cnty_fips" layer:\n{}'
                   .format(regs.loc[~fips_check, 'FIPS']))
            logger.error(msg)
            raise RuntimeError(msg)

        regs = fips_df.merge(regs, on='FIPS', how='right')

        return regs

    def _preflight_check(self, regs_fpath, multiplier):
        if regs_fpath:
            if multiplier:
                msg = ('A wind regulation .csv file was also provided and '
                       'will be used to determine set-back multipliers!')
                logger.warning(msg)
                warn(msg)

            multiplier = None
            regs = self._parse_regs(regs_fpath, self._excl_h5)

        elif multiplier:
            regs = None
            if isinstance(multiplier, str):
                multiplier = self.MULTIPLIERS[multiplier]

        else:
            msg = ('Computing set-backs requires either a wind regulations '
                   '.csv file or a generic multiplier!')
            logger.error(msg)
            raise RuntimeError(msg)

        return regs, multiplier

    def _rasterize_set_backs(self, set_backs):
        """
        Convert set_backs geometries into exclusions array

        Parameters
        ----------
        set_backs : geopandas.GeoDataFrame
            GeoDataFrame of buffered setback geometries. Geometries will be
            rasterized and given values of 1

        Returns
        -------
        arr : ndarray
            Exclusions layer arr with proper shape to write to self._excl_h5
        """
        shapes = []
        for _, row in set_backs.iterrows():
            shapes.append((row['geometry'], 1))

        arr = features.rasterize(shapes=shapes, out_shape=self._shape, fill=0,
                                 transform=self._profile['transform'],
                                 dtype='uint8')

        return np.expand_dims(arr, axis=0)

    def _write_layer(self, layer, arr, description=None):
        """
        Write exclusion layer to disc

        Parameters
        ----------
        layer : str
            Exclusion layer name (dataset name)
        arr : ndarray
            Exclusion layers array
        description : str
            Description of exclusion layer (set as an attribute)
        """
        with h5py.File(self._excl_h5, mode='a') as f:
            ds = f.create_dataset(layer, shape=arr.shape,
                                  dtype=arr.dtype, chunks=self._chunks,
                                  data=arr)
            logger.debug('\t- {} created and loaded'.format(layer))
            ds.attrs['profile'] = json.dumps(self._profile)
            if description is not None:
                ds.attrs['description'] = description
                logger.debug('\t- Description for {} added:\n{}'
                             .format(layer, description))
