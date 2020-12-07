# -*- coding: utf-8 -*-
"""
Handler to convert exclusion to/from .h5 and .geotiff
"""
from abc import ABC, abstractclassmethod, abstractstaticmethod
from concurrent.futures import as_completed
import geopandas as gpd
import h5py
import json
import logging
import numpy as np
import os
from rasterio import features
import re
from shapely.geometry import shape
from warnings import warn

from rex.utilities import parse_table, SpawnProcessPool
from reV.handlers.exclusions import ExclusionLayers

logger = logging.getLogger(__name__)


class BaseWindSetBacks(ABC):
    """
    Create exclusions layers for wind setbacks
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
            blade tip height which is used to determine setback distance
        rotor_diameter : float | int
            Turbine rotor diameter (m), used along with hub height to compute
            blade tip height which is used to determine setback distance
        regs_fpath : str | None, optional
            Path to wind regulations .csv file, if None create global
            setbacks, by default None
        multiplier : int | float | str | None, optional
            setback multiplier to use if wind regulations are not supplied,
            if str, must one of {'high': 3, 'moderate': 1.1},
            by default None
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        chunks : tuple, optional
            Chunk size to use for setback layers, if None use default chunk
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
    def generic_setback(self):
        """
        Default setback of turbine tip height * multiplier, used for global
        setbacks

        Returns
        -------
        float
        """
        if self._multi is None:
            setback = None
        else:
            setback = self.tip_height * self._multi

        return setback

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
        """
        Parse wind regulations, combine with county geometries from
        exclusions .h5 file. The county geometries are intersected with
        features to compute county specific setbacks.

        Parameters
        ----------
        regs_fpath : str
            Path to wind regulations .csv file, must have a county FIPS column
            to match with "cnty_fips" exclusion layer.
        excl_h5 : str
            Path to .h5 file containing or to contain exclusion layers,
            one layer must be 'cnty_fips'

        Returns
        -------
        regs: geopandas.GeoDataFrame
            GeoDataFrame with county level wind setback regulations merged
            with county geometries, use for intersecting with setback features
        """
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

    @staticmethod
    def _get_setback(cnty_regs, tip_height, rotor_diameter):
        """
        Compute the setback distance in meters from the county regulations,
        turbine tip height or rotor diameter

        Parameters
        ----------
        cnty_regs : pandas.Series
            Pandas Series with wind regulations for a single county / feature
            type
        tip_height : int | float
            Turbine tip height, used to compute setbacks from hub-height
            multiplier regulations
        rotor_diameter : int | float
            Turbine rotor diameter, used to compute setbacks from
            rotor-diameter multiplier regulations

        Returns
        -------
        setback : float | None
            setback distance in meters, None if the setback "Value Type"
            was not recognized
        """
        setback_type = cnty_regs['Value Type']
        setback = cnty_regs['Value']
        if setback_type == ' Hub-height Multiplier':
            setback *= tip_height
        elif setback_type == 'Rotor-Diameter Multiplier':
            setback_type *= rotor_diameter
        elif setback_type != 'Meters':
            msg = ('Cannot create setback for {}, expecting '
                   '"Hub-height Multiplier", '
                   '"Rotor-Diameter Multiplier", or '
                   '"Meters", but got {}'
                   .format(cnty_regs['County'], setback_type))
            logger.warning(msg)
            warn(msg)
            setback = None

        return setback

    @abstractclassmethod
    def _existing_state_setbacks(cls, state_features, state_regs,
                                 tip_height, rotor_diameter):
        """
        Abstract method to compute existing state setbacks

        Parameters
        ----------
        state_features : geopandas.GeoDataFrame
            Features to setback from
        state_regs : pandas.DataFrame
            Wind regulations for same state as geojson
        tip_height : float
            Turbine blade tip height in meters
        rotor_diameter : float
            Turbine rotor diameter in meters

        Returns
        -------
        setbacks : list
            List of setback geometries for given state
        """
        setbacks = []
        for _, cnty in state_regs.iterrows():
            setback = cls._get_setback(cnty, tip_height, rotor_diameter)
            if setback is not None:
                tmp = gpd.sjoin(state_features, cnty, how='inner',
                                op='intersects')
                tmp['geometry'] = tmp.buffer(setback)

                setbacks.extend((geom, 1) for geom in tmp['geometry'])

        return setbacks

    @abstractstaticmethod
    def _general_state_setbacks(state_features, setback):
        """
        Abstract method to compute general state setbacks

        Parameters
        ----------
        state_features : geopandas.GeoDataFrame
            Features to setback from
        setback : float
            Generic set back distance in meters

        Returns
        -------
        setbacks : list
            List of setback geometries for given state
        """
        state_features['geometry'] = state_features.buffer(setback)
        setbacks = [(geom, 1) for geom in state_features['geometry']]

        return setbacks

    def _preflight_check(self, regs_fpath, multiplier):
        """
        Run preflight checks on WindSetBack inputs:
        1) Ensure either a wind regulations .csv is provided, or
           a setback multiplier
        2) Ensure wind regulations has county FIPS, map regulations to county
           geometries from exclusions .h5 file
        3) Ensure multiplier is a valid entry, either a float or one of
           {'high': 3, 'moderate': 1.1}

        Parameters
        ----------
        regs_fpath : str | None
            Path to wind regulations .csv file, if None create global
            setbacks
        multiplier : int | float | str | None
            setback multiplier to use if wind regulations are not supplied,
            if str, must one of {'high': 3, 'moderate': 1.1}

        Returns
        -------
        regs : geopands.GeoDataFrame | None
            Wind Regulations
        Multiplier : float | None
            Generic setbacks multiplier
        """
        if regs_fpath:
            if multiplier:
                msg = ('A wind regulation .csv file was also provided and '
                       'will be used to determine setback multipliers!')
                logger.warning(msg)
                warn(msg)

            multiplier = None
            regs = self._parse_regs(regs_fpath, self._excl_h5)

        elif multiplier:
            regs = None
            if isinstance(multiplier, str):
                multiplier = self.MULTIPLIERS[multiplier]

        else:
            msg = ('Computing setbacks requires either a wind regulations '
                   '.csv file or a generic multiplier!')
            logger.error(msg)
            raise RuntimeError(msg)

        return regs, multiplier

    def _rasterize_setbacks(self, shapes):
        """
        Convert setbacks geometries into exclusions array

        Parameters
        ----------
        setbacks : list
            List of (geometry, 1) pairs to rasterize. Each geometry is a
            feature buffered by the desired setback distance in meters

        Returns
        -------
        arr : ndarray
            Exclusions layer arr with proper shape to write to self._excl_h5
        """
        arr = features.rasterize(shapes=shapes, out_shape=self._shape, fill=0,
                                 transform=self._profile['transform'],
                                 dtype='uint8')

        return np.expand_dims(arr, axis=0)

    def _compute_general_setbacks(self, features, *args, max_workers=None):
        """
        Compute general setbacks for all states either in serial or parallel

        Parameters
        ----------
        features : list
            list of feature files to compute setbacks from
        args : obj
            Additional arguments to pass to _general_state_setbacks
            along with each entry in the features list
        max_workers : int, optional
            Number of workers to use for setback computation, if 1 run in
            serial, if > 1 run in parallel with that many workers, if None
            run in parallel on all available cores, by default None

        Returns
        -------
        setbacks : ndarray
            Raster array of setbacks, ready to be written to the exclusions
            .h5 file as a new exclusion layer
        """
        if max_workers is None:
            max_workers = os.cpu_count()

        setbacks = []
        if max_workers > 1:
            logger.info('Computing general setbacks in parallel using {} '
                        'workers'.format(max_workers))
            loggers = [__name__, 'reVX']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                futures = []
                for state_features in features:
                    future = exe.submit(self._general_state_setbacks,
                                        state_features, *args)
                    futures.append(future)

                for i, future in enumerate(as_completed(futures)):
                    setbacks.extend(future.result())
                    logger.debug('Computed setbacks for {} of {} states'
                                 .format((i + 1), len(features)))
        else:
            logger.info('Computing general setbacks in serial')
            for i, state_features in enumerate(features):
                setbacks.extend(self._general_state_setbacks(
                    state_features, *args))
                logger.debug('Computed setbacks for {} of {} states'
                             .format((i + 1), len(features)))

        setbacks = self._rasterize_setbacks(setbacks)

        return setbacks

    def _write_layer(self, layer, arr, description=None):
        """
        Write exclusion layer to disc

        Parameters
        ----------
        layer : str
            Exclusion layer name (dataset name)
        arr : ndarray
            Exclusion layers array
        description : str, optional
            Description of exclusion layer (set as an attribute),
            by default None
        """
        if self._hsds:
            msg = ('Cannot write new layers to an exclusion file hosted in '
                   'the cloud behind HSDS!')
            logger.error(msg)
            raise RuntimeError(msg)

        with h5py.File(self._excl_h5, mode='a') as f:
            ds = f.create_dataset(layer,
                                  shape=arr.shape,
                                  dtype=arr.dtype,
                                  chunks=(1, ) + self._chunks,
                                  data=arr)
            logger.debug('\t- {} created and loaded'.format(layer))
            ds.attrs['profile'] = json.dumps(self._profile)
            if description is not None:
                ds.attrs['description'] = description
                logger.debug('\t- Description for {} added:\n{}'
                             .format(layer, description))


class StructureWindSetBacks(BaseWindSetBacks):
    """
    Structure Wind setbacks
    """
    @staticmethod
    def _split_state_name(state_name):
        """
        Split state name at capitals to map .geojson files to regulations
        state names

        Parameters
        ----------
        state_name : str
            State name from geojson files paths with out spaces

        Returns
        -------
        str
            State names with spaces added between Capitals (names) to match
            wind regulations state names
        """
        state_name = ' '.join(a for a
                              in re.split(r'([A-Z][a-z]*)', state_name)
                              if a)

        return state_name

    @classmethod
    def _map_structures_dir(cls, structure_dir):
        """
        Map structures .geojson files to state based on file name

        Parameters
        ----------
        structure_dir : str
            Path to directory containing microsoft strucutes *.geojsons. Used
            to identify structures to build setbacks from. Files should be
            by state

        Returns
        -------
        structure_state_map : dict
            Dictionary mapping state to geojson file path
        """
        structure_state_map = {}
        for file in os.listdir(structure_dir):
            if file.endswith('.geojson'):
                state = file.split('.')[0]
                state = cls._split_state_name(state)
                structure_state_map[state] = os.path.join(structure_dir, file)

        return structure_state_map

    @staticmethod
    def _parse_structures(structure_fpath, crs):
        """
        Load structures from geojson, convert to exclusions coordinate system

        Parameters
        ----------
        structure_fpath : str
            Path to Microsoft .geojson of structures in a given state
        crs : str
            Coordinate reference system to convert structures geometries into

        Returns
        -------
        structures : geopandas.GeoDataFrame
            Geometries for structures in geojson, in exclusion coordinate
            system
        """
        structures = gpd.read_file(structure_fpath)

        return structures.to_crs(crs)

    @classmethod
    def _existing_state_setbacks(cls, structure_fpath, crs, state_regs,
                                 tip_height, rotor_diameter):
        """
        Compute existing state structure setbacks

        Parameters
        ----------
        structure_fpath : str
            Path to Microsoft .geojson of structures in a given state
        crs : str
            Coordinate reference system to convert structures geometries into
        state_regs : pandas.DataFrame
            Wind regulations for same state as geojson
        tip_height : float
            Turbine blade tip height in meters
        rotor_diameter : float
            Turbine rotor diameter in meters

        Returns
        -------
        setbacks : list
            List of setback geometries for given state
        """
        structures = cls._parse_structures(structure_fpath, crs)
        setbacks = super()._existing_state_setbacks(structures,
                                                    state_regs,
                                                    tip_height,
                                                    rotor_diameter)

        return setbacks

    @classmethod
    def _general_state_setbacks(cls, structure_fpath, crs, setback):
        """
        Compute general state setbacks

        Parameters
        ----------
        structure_fpath : str
            Path to Microsoft .geojson of structures in a given state
        crs : str
            Coordinate reference system to convert structures geometries into
        setback : float
            Generic set back distance in meters

        Returns
        -------
        setbacks : list
            List of setback geometries for given state
        """
        structures = cls._parse_structures(structure_fpath, crs)
        setbacks = super()._general_state_setbacks(structures, setback)

        return setbacks

    def _compute_existing_setbacks(self, structures_state_map,
                                   max_workers=None):
        """
        Compute existing setbacks for all states either in serial or parallel

        Parameters
        ----------
        structures_state_map : list
            Dictionary mapping structure .geojson files to state names
        max_workers : int, optional
            Number of workers to use for setback computation, if 1 run in
            serial, if > 1 run in parallel with that many workers, if None
            run in parallel on all available cores, by default None

        Returns
        -------
        setbacks : ndarray
            Raster array of setbacks, ready to be written to the exclusions
            .h5 file as a new exclusion layer
        """
        if max_workers is None:
            max_workers = os.cpu_count()

        regs = self._regs.groupby('State')
        crs = self._profile['crs']

        setbacks = []
        if max_workers > 1:
            logger.info('Computing existing setbacks in parallel using {} '
                        'workers'.format(max_workers))
            loggers = [__name__, 'reVX']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                futures = []
                for state, state_regs in regs:
                    structure_fpath = structures_state_map[state]
                    future = exe.submit(self._existing_state_setbacks,
                                        structure_fpath, crs, state_regs,
                                        self.tip_height, self.rotor_diameter)
                    futures.append(future)

                for i, future in enumerate(as_completed(futures)):
                    setbacks.extend(future.result())
                    logger.debug('Computed setbacks for {} of {} states'
                                 .format((i + 1), len(regs)))
        else:
            logger.info('Computing existing setbacks in serial')
            for i, (state, state_regs) in enumerate(regs):
                structure_fpath = structures_state_map[state]
                setbacks.extend(self._existing_state_setbacks(
                    structure_fpath, crs, state_regs,
                    self.tip_height, self.rotor_diameter))
                logger.debug('Computed setbacks for {} of {} states'
                             .format((i + 1), len(regs)))

        setbacks = self._rasterize_setbacks(setbacks)

        return setbacks

    def compute_setbacks(self, structures_dir, max_workers=None,
                         layer=None, description=None):
        """
        Compute setbacks for all states either in serial or parallel.
        Existing setbacks are computed if a wind regulations file was supplied
        during class initialization, otherwise general setbacks are computed

        Parameters
        ----------
        structure_dir : str
            Path to directory containing microsoft strucutes *.geojsons. Used
            to identify structures to build setbacks from. Files should be
            by state
        max_workers : int, optional
            Number of workers to use for setback computation, if 1 run in
            serial, if > 1 run in parallel with that many workers, if None
            run in parallel on all available cores, by default None
        layer : str, optional
            Name of new layer to write to exclusions .h5 file containing
            computed setbacks, if None do not write to disc, by default None
        description : str, optional
            Description of exclusion layer (set as an attribute),
            by default None

        Returns
        -------
        setbacks : ndarray
            Raster array of setbacks, ready to be written to the exclusions
            .h5 file as a new exclusion layer
        """
        structures_state_map = self._map_structures_dir(structures_dir)
        if self._regs is not None:
            setbacks = self._compute_existing_setbacks(structures_state_map,
                                                       max_workers=max_workers)
        else:
            crs = self._profile['crs']
            setbacks = self._compute_general_setbacks(
                structures_state_map.values(), crs, self.generic_setback,
                max_workers=max_workers)

        if layer is not None:
            self._write_layer(layer, setbacks, description=description)

        return setbacks

    @classmethod
    def run(cls, excl_h5, structures_dir, layer_name, hub_height,
            rotor_diameter, regs_fpath=None, multiplier=None,
            chunks=(128, 128), max_workers=None, description=None):
        """
        Compute structural setbacks and write them as a new layer to the
        exclusions .h5 file. If a wind regulations file is given compute
        existing setbacks, otherwise compute general setbacks using the given
        multiplier and the turbine tip-height. File must be locally on disc to
        allow for writing of new layer.

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing or to contain exclusion layers
        structure_dir : str
            Path to directory containing microsoft strucutes *.geojsons. Used
            to identify structures to build setbacks from. Files should be
            by state
        layer_name : str
            Name of new layer to write to exclusions .h5 file containing
            computed setbacks
        hub_height : float | int
            Turbine hub height (m), used along with rotor diameter to compute
            blade tip height which is used to determine setback distance
        rotor_diameter : float | int
            Turbine rotor diameter (m), used along with hub height to compute
            blade tip height which is used to determine setback distance
        regs_fpath : str | None, optional
            Path to wind regulations .csv file, if None create global
            setbacks, by default None
        multiplier : int | float | str | None, optional
            setback multiplier to use if wind regulations are not supplied,
            if str, must one of {'high': 3, 'moderate': 1.1},
            by default None
        chunks : tuple, optional
            Chunk size to use for setback layers, if None use default chunk
            size in excl_h5, by default (128, 128)
        max_workers : int, optional
            Number of workers to use for setback computation, if 1 run in
            serial, if > 1 run in parallel with that many workers, if None
            run in parallel on all available cores, by default None
        description : str, optional
            Description of exclusion layer (set as an attribute),
            by default None
        """
        setbacks = cls(excl_h5, hub_height, rotor_diameter,
                       regs_fpath=regs_fpath, multiplier=multiplier,
                       hsds=False, chunk=chunks)
        setbacks.compute_setbacks(structures_dir, layer=layer_name,
                                  max_workers=max_workers,
                                  description=description)
