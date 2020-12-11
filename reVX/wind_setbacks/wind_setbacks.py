# -*- coding: utf-8 -*-
"""
Handler to convert exclusion to/from .h5 and .geotiff
"""
from abc import ABC, abstractstaticmethod
from concurrent.futures import as_completed
import fiona
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


class BaseWindSetbacks(ABC):
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
            Path to .h5 file containing exclusion layers, will also be the
            location of any new setback layers
        hub_height : float | int
            Turbine hub height (m), used along with rotor diameter to compute
            blade tip height which is used to determine setback distance
        rotor_diameter : float | int
            Turbine rotor diameter (m), used along with hub height to compute
            blade tip height which is used to determine setback distance
        regs_fpath : str | None, optional
            Path to wind regulations .csv file, if None create generic
            setbacks using max-tip height * "multiplier", by default None
        multiplier : int | float | str | None, optional
            setback multiplier to use if wind regulations are not supplied,
            if str, must a key in {'high': 3, 'moderate': 1.1}, if supplied
            along with regs_fpath, will be ignored, multiplied with
            max-tip height, by default None
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

    @staticmethod
    def _parse_excl_properties(excl_h5, chunks, hsds=False):
        """
        Parse exclusions shape, chunk size, and profile from excl_h5 file

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing exclusion layers, will also be the
            location of any new setback layers
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

        logger.debug('Exclusions properties:\n'
                     'shape : {}\n'
                     'chunks : {}\n'
                     'profile : {}\n'
                     .format(shape, chunks, profile))

        return shape, chunks, profile

    @abstractstaticmethod
    def _parse_regs(regs_fpath):
        """
        Parse wind regulations

        Parameters
        ----------
        regs_fpath : str
            Path to wind regulations .csv file

        Returns
        -------
        regs : pandas.DataFrame
            Wind regulations table
        """
        regs = parse_table(regs_fpath)
        if 'FIPS' not in regs:
            msg = ('Wind regulations does not have county FIPS! Please add a '
                   '"FIPS" columns with the unique county FIPS values.')
            logger.error(msg)
            raise RuntimeError(msg)

        return regs

    @classmethod
    def _parse_county_regs(cls, regs_fpath, excl_h5):
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
            Path to .h5 file containing exclusion layers, will also be the
            location of any new setback layers, one layer must be 'cnty_fips'

        Returns
        -------
        regs: geopandas.GeoDataFrame
            GeoDataFrame with county level wind setback regulations merged
            with county geometries, use for intersecting with setback features
        """
        regs = cls._parse_regs(regs_fpath)

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
        logger.debug('Wind regulations were provided for {} counties'
                     .format(len(regs)))

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
        if setback_type == 'Max-tip Height Multiplier':
            setback *= tip_height
        elif setback_type == 'Rotor-Diameter Multiplier':
            setback *= rotor_diameter
        elif setback_type != 'Meters':
            msg = ('Cannot create setback for {}, expecting '
                   '"Max-tip Height Multiplier", '
                   '"Rotor-Diameter Multiplier", or '
                   '"Meters", but got {}'
                   .format(cnty_regs['County'], setback_type))
            logger.warning(msg)
            warn(msg)
            setback = None

        return setback

    @abstractstaticmethod
    def _parse_features(features_fpath, crs):
        """
        Abstract method to parse features

        Parameters
        ----------
        features_fpath : str
            Path to file containing features to setback from
        crs : str
            Coordinate reference system to convert structures geometries into

        Returns
        -------
        features : geopandas.GeoDataFrame
            Geometries of features to setback from in exclusion coordinate
            system
        """

    @classmethod
    def _map_features_dir(cls, features_dir):
        """
        Map features files to state based on file name

        Parameters
        ----------
        features_dir : str
            Path to directory containing features files. Used
            to identify features to build setbacks from. Files should be
            by state

        Returns
        -------
        features_state_map : dict
            Dictionary mapping state to features file path
        """

    @classmethod
    def _compute_generic_setbacks(cls, features_fpath, crs, setback):
        """
        Abstract method to compute generic setbacks

        Parameters
        ----------
        features_fpath : str
            Path to file containing features to setback from
        crs : str
            Coordinate reference system to convert structures geometries into
        setback : float
            Generic set back distance in meters

        Returns
        -------
        setbacks : list
            List of setback geometries for given features
        """
        features = cls._parse_features(features_fpath, crs)

        features['geometry'] = features.buffer(setback)
        setbacks = [(geom, 1) for geom in features['geometry']]

        return setbacks

    @classmethod
    def _compute_local_setbacks(cls, features_fpath, crs, wind_regs,
                                tip_height, rotor_diameter):
        """
        Compute local features setbacks

        Parameters
        ----------
        features_fpath : str
            Path to file containg feature to setback from
        crs : str
            Coordinate reference system to convert structures geometries into
        wind_regs : pandas.DataFrame
            Wind regulations that define setbacks by county
        tip_height : float
            Turbine blade tip height in meters
        rotor_diameter : float
            Turbine rotor diameter in meters

        Returns
        -------
        setbacks : list
            List of setback geometries
        """
        features = cls._parse_features(features_fpath, crs)

        setbacks = []
        for i in range(len(wind_regs)):
            cnty = wind_regs.iloc[[i]]
            setback = cls._get_setback(cnty.iloc[0], tip_height,
                                       rotor_diameter)
            if setback is not None:
                logger.debug('- Computing setbacks for county FIPS {}'
                             .format(cnty.iloc[0]['FIPS']))
                tmp = gpd.sjoin(features, cnty,
                                how='inner', op='intersects')
                tmp['geometry'] = tmp.buffer(setback)

                setbacks.extend((geom, 1) for geom in tmp['geometry'])

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
            regs = self._parse_county_regs(regs_fpath, self._excl_h5)
            logger.debug('Computing setbacks using regulations provided in: {}'
                         .format(regs_fpath))
        elif multiplier:

            regs = None
            if isinstance(multiplier, str):
                multiplier = self.MULTIPLIERS[multiplier]

            logger.debug('Computing setbacks using generic Max-tip Height '
                         'Multiplier of {}'.format(multiplier))
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
        logger.debug('Generating setbacks exclusion array of shape {}'
                     .format(self._shape))
        arr = features.rasterize(shapes=shapes,
                                 out_shape=self._shape[1:],
                                 fill=0,
                                 transform=self._profile['transform'],
                                 dtype='uint8')

        return np.expand_dims(arr, axis=0)

    def _generic_setbacks(self, features, max_workers=None):
        """
        Compute generic setbacks for all states either in serial or parallel

        Parameters
        ----------
        features : list
            list of feature files to compute setbacks from
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

        crs = self._profile['crs']

        setbacks = []
        if max_workers > 1:
            logger.info('Computing generic setbacks in parallel using {} '
                        'workers'.format(max_workers))
            loggers = [__name__, 'reVX']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                futures = []
                for state_features in features:
                    future = exe.submit(self._compute_generic_setbacks,
                                        state_features, crs,
                                        self.generic_setback)
                    futures.append(future)

                for i, future in enumerate(as_completed(futures)):
                    setbacks.extend(future.result())
                    logger.debug('Computed setbacks for {} of {} states'
                                 .format((i + 1), len(features)))
        else:
            logger.info('Computing generic setbacks in serial')
            for i, state_features in enumerate(features):
                setbacks.extend(self._compute_generic_setbacks(
                    state_features, crs, self.generic_setback))
                logger.debug('Computed setbacks for {} of {} states'
                             .format((i + 1), len(features)))

        setbacks = self._rasterize_setbacks(setbacks)

        return setbacks

    def _local_setbacks(self, features_state_map, groupby,
                        max_workers=None):
        """
        Compute local setbacks for all states either in serial or parallel.
        Setbacks are based on local regulation supplied in the wind regulations
        .csv file

        Parameters
        ----------
        features_state_map : dict
            Dictionary mapping features files to states
        groupby : str
            Column to groupby regulations on and map to features files,
            for structures typically "State", for roads typically "Abbr"
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

        regs = self._regs.groupby(groupby)
        crs = self._profile['crs']

        setbacks = []
        if max_workers > 1:
            logger.info('Computing local setbacks in parallel using {} '
                        'workers'.format(max_workers))
            loggers = [__name__, 'reVX']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                futures = []
                for state, state_regs in regs:
                    if state in features_state_map:
                        features_fpath = features_state_map[state]
                        future = exe.submit(self._compute_local_setbacks,
                                            features_fpath, crs, state_regs,
                                            self.tip_height,
                                            self.rotor_diameter)
                        futures.append(future)

                for i, future in enumerate(as_completed(futures)):
                    setbacks.extend(future.result())
                    logger.debug('Computed setbacks for {} of {} states'
                                 .format((i + 1), len(futures)))
        else:
            logger.info('Computing local setbacks in serial')
            for i, (state, state_regs) in enumerate(regs):
                if state in features_state_map:
                    features_fpath = features_state_map[state]
                    setbacks.extend(self._compute_local_setbacks(
                        features_fpath, crs, state_regs,
                        self.tip_height, self.rotor_diameter))
                    logger.debug('Computed setbacks for {} of {} states'
                                 .format((i + 1), len(regs)))

        setbacks = self._rasterize_setbacks(setbacks)

        return setbacks

    def _write_layer(self, layer, arr, description=None, replace=False):
        """
        Write exclusion layer to exclusions .h5 file

        Parameters
        ----------
        layer : str
            Exclusion layer name (dataset name)
        arr : ndarray
            Exclusion layers array
        description : str, optional
            Description of exclusion layer (set as an attribute),
            by default None
        replace : bool, optional
            Flag to replace local layer data with arr if layer already
            exists in the exlcusion .h5 file, by default False
        """
        if self._hsds:
            msg = ('Cannot write new layers to an exclusion file hosted in '
                   'the cloud behind HSDS!')
            logger.error(msg)
            raise RuntimeError(msg)

        with h5py.File(self._excl_h5, mode='a') as f:
            if layer in f:
                msg = "{} is already present in {}".format(layer,
                                                           self._excl_h5)
                if replace:
                    msg += ', layer data will be replaced with new setbacks'
                    logger.warning(msg)
                    warn(msg)

                    f[layer][...] = arr
                else:
                    msg += ' to replace layer data, set replace=True'
                    logger.error(msg)
                    raise RuntimeError(msg)
            else:
                ds = f.create_dataset(layer,
                                      shape=arr.shape,
                                      dtype=arr.dtype,
                                      chunks=self._chunks,
                                      data=arr)
                logger.debug('\t- {} created and loaded'.format(layer))
                ds.attrs['profile'] = json.dumps(self._profile)
                if description is not None:
                    ds.attrs['description'] = description
                    logger.debug('\t- Description for {} added:\n{}'
                                 .format(layer, description))

    def compute_setbacks(self, features_dir, groupby, max_workers=None,
                         layer=None, description=None, replace=False):
        """
        Compute setbacks for all states either in serial or parallel.
        Existing setbacks are computed if a wind regulations file was supplied
        during class initialization, otherwise generic setbacks are computed

        Parameters
        ----------
        features_dir : str
            Path to directory containing features files. Used
            to identify features to build setbacks from. Files should be
            by state
        groupby : str
            Column to groupby regulations on and map to features files,
            for structures typically "State", for roads typically "Abbr"
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
        replace : bool, optional
            Flag to replace local layer data with arr if layer already
            exists in the exlcusion .h5 file, by default False

        Returns
        -------
        setbacks : ndarray
            Raster array of setbacks, ready to be written to the exclusions
            .h5 file as a new exclusion layer
        """
        features_state_map = self._map_features_dir(features_dir)
        if self._regs is not None:
            setbacks = self._local_setbacks(features_state_map, groupby,
                                            max_workers=max_workers)
        else:
            setbacks = self._generic_setbacks(features_state_map.values(),
                                              max_workers=max_workers)

        if layer is not None:
            logger.debug('Writing setbacks to {} as layer {}'
                         .format(self._excl_h5, layer))
            self._write_layer(layer, setbacks, description=description,
                              replace=replace)

        return setbacks


class StructureWindSetbacks(BaseWindSetbacks):
    """
    Structure Wind setbacks
    """
    @staticmethod
    def _parse_regs(regs_fpath):
        """
        Parse wind regulations, reduce table to just structures

        Parameters
        ----------
        regs_fpath : str
            Path to wind regulations .csv file

        Returns
        -------
        regs : pandas.DataFrame
            Wind regulations table
        """
        regs = BaseWindSetbacks._parse_regs(regs_fpath)

        mask = ((regs['Feature Type'] == 'Structures')
                & (regs['Comment'] != 'Occupied Community Buildings'))
        regs = regs.loc[mask]

        return regs

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
    def _map_features_dir(cls, structures_dir):
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
        features_state_map = {}
        for file in os.listdir(structures_dir):
            if file.endswith('.geojson'):
                state = file.split('.')[0]
                state = cls._split_state_name(state)
                features_state_map[state] = os.path.join(structures_dir, file)

        return features_state_map

    @staticmethod
    def _parse_features(structure_fpath, crs):
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
    def run(cls, excl_h5, structures_dir, layer_name, hub_height,
            rotor_diameter, regs_fpath=None, multiplier=None,
            chunks=(128, 128), max_workers=None, description=None,
            replace=False):
        """
        Compute structural setbacks and write them as a new layer to the
        exclusions .h5 file. If a wind regulations file is given compute
        local setbacks, otherwise compute generic setbacks using the given
        multiplier and the turbine tip-height. File must be locally on disc to
        allow for writing of new layer.

        Parametersß
        ----------
        excl_h5 : str
            Path to .h5 file containing exclusion layers, will also be the
            location of any new setback layers
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
            Path to wind regulations .csv file, if None create generic
            setbacks using max-tip height * "multiplier", by default None
        multiplier : int | float | str | None, optional
            setback multiplier to use if wind regulations are not supplied,
            if str, must a key in {'high': 3, 'moderate': 1.1}, if supplied
            along with regs_fpath, will be ignored, multiplied with
            max-tip height, by default None
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
        replace : bool, optional
            Flag to replace local layer data with arr if layer already
            exists in the exlcusion .h5 file, by default False
        """
        setbacks = cls(excl_h5, hub_height, rotor_diameter,
                       regs_fpath=regs_fpath, multiplier=multiplier,
                       hsds=False, chunks=chunks)
        setbacks.compute_setbacks(structures_dir, "State", layer=layer_name,
                                  max_workers=max_workers,
                                  description=description,
                                  replace=replace)


class RoadWindSetbacks(BaseWindSetbacks):
    """
    Road Wind setbacks
    """
    @staticmethod
    def _parse_regs(regs_fpath):
        """
        Parse wind regulations, reduce table to just roads

        Parameters
        ----------
        regs_fpath : str
            Path to wind regulations .csv file

        Returns
        -------
        regs : pandas.DataFrame
            Wind regulations table
        """
        regs = BaseWindSetbacks._parse_regs(regs_fpath)

        mask = regs['Feature Type'].isin(['Roads', 'Highways', 'Highways 111'])
        regs = regs.loc[mask]

        return regs

    @classmethod
    def _map_features_dir(cls, roads_dir):
        """
        Map roads .gdb files to state based on file name

        Parameters
        ----------
        roads_dir : str
            Path to directory containing here streets *.gdb files. Used
            to identify roads to build setbacks from. Files should be
            by state

        Returns
        -------
        roads_state_map : dict
            Dictionary mapping state to gdb file path
        """
        roads_state_map = {}
        for file in os.listdir(roads_dir):
            if file.endswith('.gdb') and file.startswith('Streets_USA'):
                state = file.split('.')[0].split('_')[-1]
                roads_state_map[state] = os.path.join(roads_dir, file)

        return roads_state_map

    @staticmethod
    def _parse_features(roads_fpath, crs):
        """
        Load roads from gdb file, convert to exclusions coordinate system

        Parameters
        ----------
        roads_fpath : str
            Path to here streets gdb file for given state
        crs : str
            Coordinate reference system to convert structures geometries into

        Returns
        -------
        roads : geopandas.GeoDataFrame.sindex
            Geometries for roads in gdb file, in exclusion coordinate
            system
        """
        lyr = fiona.listlayers(roads_fpath)[0]
        roads = gpd.read_file(roads_fpath, driver='FileGDB', layer=lyr)

        return roads.to_crs(crs)

    @classmethod
    def _compute_local_setbacks(cls, roads_fpath, crs, wind_regs,
                                tip_height, rotor_diameter):
        """
        Compute local road setbacks

        Parameters
        ----------
        roads_fpath : str
            Path to here streets gdb file with roads to setback from
        crs : str
            Coordinate reference system to convert structures geometries into
        wind_regs : pandas.DataFrame
            Wind regulations by county
        tip_height : float
            Turbine blade tip height in meters
        rotor_diameter : float
            Turbine rotor diameter in meters

        Returns
        -------
        setbacks : list
            List of setback geometries for given eatures
        """
        features = cls._parse_features(roads_fpath, crs)
        features = features[['StreetName', 'geometry']]
        si = features.sindex

        setbacks = []
        for i in range(len(wind_regs)):
            cnty = wind_regs.iloc[i]
            setback = cls._get_setback(cnty, tip_height,
                                       rotor_diameter)
            if setback is not None:
                logger.debug('- Computing setbacks for county FIPS {}'
                             .format(cnty['FIPS']))
                # spatial index bounding box and intersection
                bb_index = \
                    list(si.intersection(cnty['geometry'].bounds))
                bb_pos = features.iloc[bb_index]
                tmp = bb_pos.intersection(cnty['geometry'])
                tmp = gpd.GeoDataFrame(geometry=tmp)
                tmp = tmp[~tmp.is_empty]
                # Buffer setback
                tmp['geometry'] = tmp.buffer(setback)

                setbacks.extend((geom, 1) for geom in tmp['geometry'])

        return setbacks

    @classmethod
    def run(cls, excl_h5, roads_dir, layer_name, hub_height,
            rotor_diameter, regs_fpath=None, multiplier=None,
            chunks=(128, 128), max_workers=None, description=None,
            replace=False):
        """
        Compute road setbacks and write them as a new layer to the
        exclusions .h5 file. If a wind regulations file is given compute
        local setbacks, otherwise compute generic setbacks using the given
        multiplier and the turbine tip-height. File must be locally on disc to
        allow for writing of new layer.

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing exclusion layers, will also be the
            location of any new setback layers
        roads_dir : str
            Path to directory containing here streets *.gdb files. Used
            to identify roads to build setbacks from. Files should be
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
            Path to wind regulations .csv file, if None create generic
            setbacks using max-tip height * "multiplier", by default None
        multiplier : int | float | str | None, optional
            setback multiplier to use if wind regulations are not supplied,
            if str, must a key in {'high': 3, 'moderate': 1.1}, if supplied
            along with regs_fpath, will be ignored, multiplied with
            max-tip height, by default None
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
        replace : bool, optional
            Flag to replace local layer data with arr if layer already
            exists in the exlcusion .h5 file, by default False
        """
        setbacks = cls(excl_h5, hub_height, rotor_diameter,
                       regs_fpath=regs_fpath, multiplier=multiplier,
                       hsds=False, chunks=chunks)
        setbacks.compute_setbacks(roads_dir, 'Abbr', layer=layer_name,
                                  max_workers=max_workers,
                                  description=description,
                                  replace=replace)


class TransmissionWindSetbacks(BaseWindSetbacks):
    """
    Transmission Wind setbacks, computed against a single set of transmission
    features instead of against state level features
    """
    @staticmethod
    def _parse_regs(regs_fpath):
        """
        Parse wind regulations, reduce table to just transmission

        Parameters
        ----------
        regs_fpath : str
            Path to wind regulations .csv file

        Returns
        -------
        regs : pandas.DataFrame
            Wind regulations table
        """
        regs = BaseWindSetbacks._parse_regs(regs_fpath)

        mask = regs['Feature Type'] == 'Transmission'
        regs = regs.loc[mask]

        return regs

    @staticmethod
    def _parse_features(transmission_fpath, crs):
        """
        Load transmission shape file, convert to exclusions coordinate system

        Parameters
        ----------
        transmission_fpath : str
            Path to transmission shape file
        crs : str
            Coordinate reference system to convert structures geometries into

        Returns
        -------
        trans : geopandas.GeoDataFrame.sindex
            Geometries for transmission features, in exclusion coordinate
            system
        """
        trans = gpd.read_file(transmission_fpath)

        return trans.to_crs(crs)

    @classmethod
    def _compute_local_setbacks(cls, features_fpath, crs, cnty,
                                tip_height, rotor_diameter):
        """
        Compute local county setbacks

        Parameters
        ----------
        features_fpath : str
            Path to shape file with features to compute setbacks from
        crs : str
            Coordinate reference system to convert structures geometries into
        cnty : geopandas.GeoDataFrame
            Wind regulations for a single county
        tip_height : float
            Turbine blade tip height in meters
        rotor_diameter : float
            Turbine rotor diameter in meters

        Returns
        -------
        setbacks : list
            List of setback geometries for given state
        """
        features = cls._parse_features(features_fpath, crs)

        setbacks = []
        setback = cls._get_setback(cnty.iloc[0], tip_height, rotor_diameter)
        if setback is not None:
            # clip the transmission lines to county geometry
            # pylint: disable=assignment-from-no-return
            tmp = gpd.clip(features, cnty)
            tmp = tmp[~tmp.is_empty]

            # Buffer setback
            tmp['geometry'] = tmp.buffer(setback)

            setbacks.extend((geom, 1) for geom in tmp['geometry'])

        return setbacks

    def compute_setbacks(self, features_fpath, max_workers=None,
                         layer=None, description=None, replace=False):
        """
        Compute setbacks for all states either in serial or parallel.
        Existing setbacks are computed if a wind regulations file was supplied
        during class initialization, otherwise generic setbacks are computed

        Parameters
        ----------
        features_fpath : str
            Path to shape file with features to compute setbacks from
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
        replace : bool, optional
            Flag to replace local layer data with arr if layer already
            exists in the exlcusion .h5 file, by default False

        Returns
        -------
        setbacks : ndarray
            Raster array of setbacks, ready to be written to the exclusions
            .h5 file as a new exclusion layer
        """
        crs = self._profile['crs']
        setbacks = []
        if self._regs is not None:
            if max_workers is None:
                max_workers = os.cpu_count()

            if max_workers > 1:
                logger.info('Computing local setbacks in parallel using {} '
                            'workers'.format(max_workers))
                loggers = [__name__, 'reVX']
                with SpawnProcessPool(max_workers=max_workers,
                                      loggers=loggers) as exe:
                    futures = []
                    for i in range(len(self._regs)):
                        cnty = self._regs.iloc[[i]]
                        future = exe.submit(self._compute_local_setbacks,
                                            features_fpath, crs, cnty,
                                            self.tip_height,
                                            self.rotor_diameter)
                        futures.append(future)

                    for i, future in enumerate(as_completed(futures)):
                        setbacks.extend(future.result())
                        logger.debug('Computed setbacks for {} of {} states'
                                     .format((i + 1), len(self._regs)))
            else:
                logger.info('Computing local setbacks in serial')
                for i in range(len(self._regs)):
                    cnty = self._regs.iloc[[i]]
                    setbacks.extend(self._compute_local_setbacks(
                        features_fpath, crs, cnty, self.tip_height,
                        self.rotor_diameter))
                    logger.debug('Computed setbacks for {} of {} states'
                                 .format((i + 1), len(self._regs)))
        else:
            logger.info('Computing generic setbacks')
            setbacks.extend(self._compute_generic_setbacks(
                features_fpath, crs, self.generic_setback))

        setbacks = self._rasterize_setbacks(setbacks)

        if layer is not None:
            logger.debug('Writing setbacks to {} as layer {}'
                         .format(self._excl_h5, layer))
            self._write_layer(layer, setbacks, description=description,
                              replace=replace)

        return setbacks

    @classmethod
    def run(cls, excl_h5, features_fpath, layer_name, hub_height,
            rotor_diameter, regs_fpath=None, multiplier=None,
            chunks=(128, 128), max_workers=None, description=None,
            replace=False):
        """
        Compute setbacks and write them as a new layer to the
        exclusions .h5 file. If a wind regulations file is given compute
        local setbacks, otherwise compute generic setbacks using the given
        multiplier and the turbine tip-height. File must be locally on disc to
        allow for writing of new layer.

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing exclusion layers, will also be the
            location of any new setback layers
        features_fpath : str
            Path to shape file with transmission or rail features to compute
            setbacks from
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
            Path to wind regulations .csv file, if None create generic
            setbacks using max-tip height * "multiplier", by default None
        multiplier : int | float | str | None, optional
            setback multiplier to use if wind regulations are not supplied,
            if str, must a key in {'high': 3, 'moderate': 1.1}, if supplied
            along with regs_fpath, will be ignored, multiplied with
            max-tip height, by default None
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
        replace : bool, optional
            Flag to replace local layer data with arr if layer already
            exists in the exlcusion .h5 file, by default False
        """
        setbacks = cls(excl_h5, hub_height, rotor_diameter,
                       regs_fpath=regs_fpath, multiplier=multiplier,
                       hsds=False, chunks=chunks)
        setbacks.compute_setbacks(features_fpath, layer=layer_name,
                                  max_workers=max_workers,
                                  description=description,
                                  replace=replace)


class RailWindSetbacks(TransmissionWindSetbacks):
    """
    Rail Wind setbacks, computed against a single set of railroad features,
    instead of state level features, uses the same approach as
    TransmissionWindSetbacks
    """
    @staticmethod
    def _parse_regs(regs_fpath):
        """
        Parse wind regulations, reduce table to just rail

        Parameters
        ----------
        regs_fpath : str
            Path to wind regulations .csv file

        Returns
        -------
        regs : pandas.DataFrame
            Wind regulations table
        """
        regs = BaseWindSetbacks._parse_regs(regs_fpath)

        mask = regs['Feature Type'] == 'Railroads'
        regs = regs.loc[mask]

        return regs

    @staticmethod
    def _parse_features(rail_fpath, crs):
        """
        Load rail shape file, convert to exclusions coordinate system

        Parameters
        ----------
        rail_fpath : str
            Path to rail shape file
        crs : str
            Coordinate reference system to convert structures geometries into

        Returns
        -------
        rail : geopandas.GeoDataFrame.sindex
            Geometries for rail features, in exclusion coordinate
            system
        """
        rail = gpd.read_file(rail_fpath)

        return rail.to_crs(crs)
