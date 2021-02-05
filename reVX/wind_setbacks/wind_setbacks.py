# -*- coding: utf-8 -*-
"""
Compute wind setbacks exclusions
"""
from abc import ABC
from concurrent.futures import as_completed
import fiona
import geopandas as gpd
import logging
import os
from rasterio import features
import re
from shapely.geometry import shape
from warnings import warn

from rex.utilities import parse_table, SpawnProcessPool
from reV.handlers.exclusions import ExclusionLayers
from reVX.utilities.exclusions_converter import ExclusionsConverter

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

    @property
    def wind_regs(self):
        """
        Wind Regulations table

        Returns
        -------
        geopands.GeoDataFrame | None
        """
        return self._regs

    @property
    def multiplier(self):
        """
        Generic setback multiplier

        Returns
        -------
        int | float
        """
        return self._multi

    @property
    def arr_shape(self):
        """
        Rasterize array shape

        Returns
        -------
        tuple
        """
        return self._shape

    @property
    def profile(self):
        """
        Geotiff profile

        Returns
        -------
        dict
        """
        return self._profile

    @property
    def crs(self):
        """
        Coordinate reference system

        Returns
        -------
        str
        """
        return self.profile['crs']

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

    @staticmethod
    def _get_setback(cnty_regs, hub_height, rotor_diameter):
        """
        Compute the setback distance in meters from the county regulations,
        turbine tip height or rotor diameter

        Parameters
        ----------
        cnty_regs : pandas.Series
            Pandas Series with wind regulations for a single county / feature
            type
        hub_height : int | float
            Turbine hub-height, used to compute setbacks from hub-height
            multiplier regulations and to compute tip-height for tip-height
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
        tip_height = hub_height + rotor_diameter / 2

        setback_type = cnty_regs['Value Type']
        setback = cnty_regs['Value']
        if setback_type == 'Max-tip Height Multiplier':
            setback *= tip_height
        elif setback_type == 'Rotor-Diameter Multiplier':
            setback *= rotor_diameter
        elif setback_type == 'Hub-height Multiplier':
            setback *= hub_height
        elif setback_type != 'Meters':
            msg = ('Cannot create setback for {}, expecting '
                   '"Max-tip Height Multiplier", '
                   '"Rotor-Diameter Multiplier", '
                   '"Hub-height Multiplier", or '
                   '"Meters", but got {}'
                   .format(cnty_regs['County'], setback_type))
            logger.warning(msg)
            warn(msg)
            setback = None

        return setback

    @staticmethod
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
        features = gpd.read_file(features_fpath)

        return features.to_crs(crs=crs)

    @staticmethod
    def _compute_local_setbacks(features, cnty, setback):
        """
        Compute local features setbacks

        Parameters
        ----------
        features : geopandas.GeoDataFrame
            Features to setback from
        cnty : geopandas.GeoDataFrame
            Wind regulations for a single county
        setback : int
            Setback distance in meters

        Returns
        -------
        setbacks : list
            List of setback geometries
        """
        logger.debug('- Computing setbacks for county FIPS {}'
                     .format(cnty.iloc[0]['FIPS']))
        mask = features.centroid.within(cnty['geometry'].values[0])
        tmp = features.loc[mask]
        tmp.loc[:, 'geometry'] = tmp.buffer(setback)

        setbacks = [(geom, 1) for geom in tmp['geometry']]

        return setbacks

    def _parse_county_regs(self, regs):
        """
        Parse wind regulations, combine with county geometries from
        exclusions .h5 file. The county geometries are intersected with
        features to compute county specific setbacks.

        Parameters
        ----------
        regs : pandas.DataFrame
            Wind regulations table

        Returns
        -------
        regs: geopandas.GeoDataFrame
            GeoDataFrame with county level wind setback regulations merged
            with county geometries, use for intersecting with setback features
        """
        if 'FIPS' not in regs:
            msg = ('Wind regulations does not have county FIPS! Please add a '
                   '"FIPS" columns with the unique county FIPS values.')
            logger.error(msg)
            raise RuntimeError(msg)

        if 'geometry' not in regs:
            regs['geometry'] = None

        regs = gpd.GeoDataFrame(regs, crs=self.crs, geometry='geometry')
        regs = regs.set_index('FIPS')

        logger.info('Merging county geometries w/ local wind '
                    'regulations')
        with ExclusionLayers(self._excl_h5) as exc:
            fips = exc['cnty_fips']
            profile = exc.get_layer_profile('cnty_fips')

        tr = profile['transform']

        for p, v in features.shapes(fips, transform=tr):
            v = int(v)
            if v in regs.index:
                regs.at[v, 'geometry'] = shape(p)

        return regs.reset_index()

    def _parse_regs(self, regs_fpath):
        """
        Parse wind regulations

        Parameters
        ----------
        regs_fpath : str
            Path to wind regulations .csv file

        Returns
        -------
        regs: geopandas.GeoDataFrame
            GeoDataFrame with county level wind setback regulations merged
            with county geometries, use for intersecting with setback features
        """
        try:
            regs = parse_table(regs_fpath)
            regs = self._parse_county_regs(regs)

            out_path = regs_fpath.split('.')[0] + '.gpkg'
            logger.debug('Saving wind regulations with county geometries as: '
                         '{}'.format(out_path))
            regs.to_file(out_path, driver='GPKG')
        except ValueError:
            regs = gpd.read_file(regs_fpath)

        fips_check = regs['geometry'].isnull()
        if fips_check.any():
            msg = ('The following county FIPS were requested in the '
                   'wind regulations but were not availble in the '
                   'Exclusions "cnty_fips" layer:\n{}'
                   .format(regs.loc[fips_check, 'FIPS']))
            logger.error(msg)
            raise RuntimeError(msg)

        return regs.to_crs(crs=self.crs)

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
        regs: geopandas.GeoDataFrame | None
            GeoDataFrame with county level wind setback regulations merged
            with county geometries, use for intersecting with setback features
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
            regs = self._parse_regs(regs_fpath)
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

    def _check_regs(self, features_fpath):  # pylint: disable=unused-argument
        """
        Reduce regs to state corresponding to features_fpath if needed

        Parameters
        ----------
        features_fpath : str
            Path to shape file with features to compute setbacks from

        Returns
        -------
        regs : geopands.GeoDataFrame | None
            Wind Regulations
        """
        logger.debug('Computing setbacks for wind regulations in {} counties'
                     .format(len(self.wind_regs)))

        return self.wind_regs

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
            Rasterized array of setbacks
        """
        logger.debug('Generating setbacks exclusion array of shape {}'
                     .format(self.arr_shape))
        arr = features.rasterize(shapes=shapes,
                                 out_shape=self.arr_shape[1:],
                                 fill=0,
                                 transform=self.profile['transform'],
                                 dtype='uint8')

        return arr

    def _write_setbacks(self, geotiff, setbacks, replace=False):
        """
        Write setbacks to geotiff, replace if requested

        Parameters
        ----------
        geotiff : str
            Path to geotiff file to save setbacks too
        setbacks : ndarray
            Rasterized array of setbacks
        replace : bool, optional
            Flag to replace local layer data with arr if layer already
            exists in the exlcusion .h5 file, by default False
        """
        if os.path.exists(geotiff):
            if not replace:
                msg = ('{} already exists. To replace it set "replace=True"'
                       .format(geotiff))
                logger.error(msg)
                raise IOError(msg)
            else:
                msg = ('{} already exists and will be replaced!'
                       .format(geotiff))
                logger.warning(msg)
                warn(msg)

        ExclusionsConverter._write_geotiff(geotiff, self.profile, setbacks)

    def compute_local_setbacks(self, features_fpath, max_workers=None):
        """
        Compute local setbacks for all counties either in serial or parallel.

        Parameters
        ----------
        features_fpath : str
            Path to shape file with features to compute setbacks from
        max_workers : int, optional
            Number of workers to use for setback computation, if 1 run in
            serial, if > 1 run in parallel with that many workers, if None
            run in parallel on all available cores, by default None

        Returns
        -------
        setbacks : ndarray
            Raster array of setbacks
        """
        wind_regs = self._check_regs(features_fpath)
        setbacks = []
        features = self._parse_features(features_fpath, self.crs)
        if max_workers is None:
            max_workers = os.cpu_count()

        if max_workers > 1:
            logger.info('Computing local setbacks in parallel using {} '
                        'workers'.format(max_workers))
            loggers = [__name__, 'reVX']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                futures = []
                for i in range(len(wind_regs)):
                    cnty = wind_regs.iloc[[i]]
                    setback = self._get_setback(cnty.iloc[0], self.hub_height,
                                                self.rotor_diameter)
                    if setback is not None:
                        idx = features.sindex.intersection(cnty.total_bounds)
                        cnty_feats = features.iloc[list(idx)]
                        future = exe.submit(self._compute_local_setbacks,
                                            cnty_feats, cnty, setback)
                        futures.append(future)

                for i, future in enumerate(as_completed(futures)):
                    setbacks.extend(future.result())
                    logger.debug('Computed setbacks for {} of {} counties'
                                 .format((i + 1), len(wind_regs)))
        else:
            logger.info('Computing local setbacks in serial')
            for i in range(len(wind_regs)):
                cnty = wind_regs.iloc[[i]]
                setback = self._get_setback(cnty.iloc[0], self.hub_height,
                                            self.rotor_diameter)
                if setback is not None:
                    idx = features.sindex.intersection(cnty.total_bounds)
                    cnty_feats = features.iloc[list(idx)]
                    setbacks.extend(self._compute_local_setbacks(cnty_feats,
                                                                 cnty,
                                                                 setback))
                    logger.debug('Computed setbacks for {} of {} counties'
                                 .format((i + 1), len(wind_regs)))

        return self._rasterize_setbacks(setbacks)

    def compute_generic_setbacks(self, features_fpath):
        """
        Compute generic setbacks.

        Parameters
        ----------
        features_fpath : str
            Path to shape file with features to compute setbacks from

        Returns
        -------
        setbacks : ndarray
            Raster array of setbacks
        """
        logger.info('Computing generic setbacks')
        features = self._parse_features(features_fpath, self.crs)
        features.loc[:, 'geometry'] = features.buffer(self.generic_setback)
        setbacks = [(geom, 1) for geom in features['geometry']]

        return self._rasterize_setbacks(setbacks)

    def compute_setbacks(self, features_fpath, max_workers=None,
                         geotiff=None, replace=False):
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
        geotiff : str, optional
            Path to save geotiff containing rasterized setbacks,
            by default None
        replace : bool, optional
            Flag to replace geotiff if it already exists, by default False

        Returns
        -------
        setbacks : ndarray
            Raster array of setbacks
        """
        if self._regs is not None:
            setbacks = self.compute_local_setbacks(features_fpath,
                                                   max_workers=max_workers)
        else:
            setbacks = self.compute_generic_setbacks(features_fpath)

        if geotiff is not None:
            logger.debug('Writing setbacks to {}'.format(geotiff))
            self._write_setbacks(geotiff, setbacks, replace=replace)

        return setbacks


class StructureWindSetbacks(BaseWindSetbacks):
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

    @staticmethod
    def _get_feature_paths(structures_path):
        """
        Find all structures .geojson files in structures dir

        Parameters
        ----------
        structure_path : str
            Path to structures geojson for a single state, or directory
            containing geojsons for all states. Used to identify structures to
            build setbacks from. Files should be by state

        Returns
        -------
        file_paths : list
            List of file paths to all structures .geojson files in
            structures_dir
        """
        if structures_path.endswith('.geojson'):
            file_paths = [structures_path]
        else:
            file_paths = []
            for file in sorted(os.listdir(structures_path)):
                if file.endswith('.geojson'):
                    file_paths.append(os.path.join(structures_path, file))

        return file_paths

    def _parse_regs(self, regs_fpath):
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
        regs = super()._parse_regs(regs_fpath)

        mask = ((regs['Feature Type'] == 'Structures')
                & (regs['Comment'] != 'Occupied Community Buildings'))
        regs = regs.loc[mask]

        return regs

    def _check_regs(self, features_fpath):
        """
        Reduce regs to state corresponding to features_fpath if needed

        Parameters
        ----------
        features_fpath : str
            Path to shape file with features to compute setbacks from

        Returns
        -------
        wind_regs : geopands.GeoDataFrame | None
            Wind Regulations
        """
        state_name = os.path.basename(features_fpath).split('.')[0]
        state = self._split_state_name(state_name)
        wind_regs = self.wind_regs
        mask = wind_regs["State"] == state

        if not mask.any():
            msg = ("There are no local regulations in {}!".format(state))
            logger.error(msg)
            raise RuntimeError(msg)

        wind_regs = wind_regs.loc[mask].reset_index(drop=True)
        logger.debug('Computing setbacks for wind regulations in {} counties'
                     .format(len(wind_regs)))

        return wind_regs

    @classmethod
    def run(cls, excl_h5, structures_path, out_dir, hub_height,
            rotor_diameter, regs_fpath=None, multiplier=None,
            chunks=(128, 128), max_workers=None, replace=False, hsds=False):
        """
        Compute state's structural setbacks and write them to a geotiff.
        If a wind regulations file is given compute local setbacks, otherwise
        compute generic setbacks using the given multiplier and the turbine
        tip-height.

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing exclusion layers, will also be the
            location of any new setback layers
        structure_path : str
            Path to structures geojson for a single state, or directory
            containing geojsons for all states.
        out_dir : str
            Directory to save setbacks geotiff(s) into
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
        replace : bool, optional
            Flag to replace geotiff if it already exists, by default False
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        """
        setbacks = cls(excl_h5, hub_height, rotor_diameter,
                       regs_fpath=regs_fpath, multiplier=multiplier,
                       hsds=hsds, chunks=chunks)

        structures_path = setbacks._get_feature_paths(structures_path)

        for fpath in structures_path:
            geotiff = os.path.basename(fpath).replace('.geojson', '.geotiff')
            geotiff = os.path.join(out_dir, geotiff)
            if os.path.exists(geotiff) and not replace:
                msg = ('{} already exists, setbacks will not be re-computed '
                       'unless replace=True'.format(geotiff))
                logger.error(msg)
            else:
                logger.info("Computing setbacks from structures in {} and "
                            "saving to {}".format(fpath, geotiff))
                setbacks.compute_setbacks(fpath, geotiff=geotiff,
                                          max_workers=max_workers,
                                          replace=replace)


class RoadWindSetbacks(BaseWindSetbacks):
    """
    Road Wind setbacks
    """
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

        return roads.to_crs(crs=crs)

    @staticmethod
    def _get_feature_paths(roads_path):
        """
        Find all roads gdb files in roads_dir

        Parameters
        ----------
        roads_path : str
            Path to state here streets gdb file or directory containing
            states gdb files. Used to identify roads to build setbacks from.
            Files should be by state

        Returns
        -------
        file_paths : list
            List of file paths to all roads .gdp files in roads_dir
        """
        if roads_path.endswith('.gdb'):
            file_paths = [roads_path]
        else:
            file_paths = []
            for file in sorted(os.listdir(roads_path)):
                if file.endswith('.gdb') and file.startswith('Streets_USA'):
                    file_paths.append(os.path.join(roads_path, file))

        return file_paths

    def _parse_regs(self, regs_fpath):
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
        regs = super()._parse_regs(regs_fpath)

        mask = regs['Feature Type'].isin(['Roads', 'Highways', 'Highways 111'])
        regs = regs.loc[mask]

        return regs

    def _check_regs(self, features_fpath):
        """
        Reduce regs to state corresponding to features_fpath if needed

        Parameters
        ----------
        features_fpath : str
            Path to shape file with features to compute setbacks from

        Returns
        -------
        wind_regs : geopands.GeoDataFrame | None
            Wind Regulations
        """
        state = features_fpath.split('.')[0].split('_')[-1]
        wind_regs = self.wind_regs
        mask = wind_regs['Abbr'] == state

        if not mask.any():
            msg = ("There are no local regulations in {}!".format(state))
            logger.error(msg)
            raise RuntimeError(msg)

        wind_regs = wind_regs.loc[mask].reset_index(drop=True)
        logger.debug('Computing setbacks for wind regulations in {} counties'
                     .format(len(wind_regs)))

        return wind_regs

    @classmethod
    def run(cls, excl_h5, roads_path, out_dir, hub_height,
            rotor_diameter, regs_fpath=None, multiplier=None,
            chunks=(128, 128), max_workers=None, replace=False, hsds=False):
        """
        Compute state's road setbacks and write them to a geotiff.
        If a wind regulations file is given compute local setbacks, otherwise
        compute generic setbacks using the given multiplier and the turbine
        tip-height.

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing exclusion layers, will also be the
            location of any new setback layers
        road_path : str
            Path to state here streets gdb file or directory containing
            states gdb files.
        out_dir : str
            Directory to save setbacks geotiff(s) into
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
        replace : bool, optional
            Flag to replace geotiff if it already exists, by default False
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        """
        setbacks = cls(excl_h5, hub_height, rotor_diameter,
                       regs_fpath=regs_fpath, multiplier=multiplier,
                       hsds=hsds, chunks=chunks)

        roads_path = setbacks._get_feature_paths(roads_path)
        for fpath in roads_path:
            geotiff = os.path.basename(fpath).replace('.gdb', '.geotiff')
            geotiff = os.path.join(out_dir, geotiff)
            if os.path.exists(geotiff) and not replace:
                msg = ('{} already exists, setbacks will not be re-computed '
                       'unless replace=True'.format(geotiff))
                logger.error(msg)
            else:
                logger.info("Computing setbacks from roads in {} and saving "
                            "to {}".format(fpath, geotiff))
                setbacks.compute_setbacks(fpath, geotiff=geotiff,
                                          max_workers=max_workers,
                                          replace=replace)


class TransmissionWindSetbacks(BaseWindSetbacks):
    """
    Transmission Wind setbacks, computed against a single set of transmission
    features instead of against state level features
    """
    @staticmethod
    def _compute_local_setbacks(features, cnty, setback):
        """
        Compute local county setbacks

        Parameters
        ----------
        features : geopandas.GeoDataFrame
            Features to setback from
        cnty : geopandas.GeoDataFrame
            Wind regulations for a single county
        setback : int
            Setback distance in meters

        Returns
        -------
        setbacks : list
            List of setback geometries
        """
        tmp = gpd.clip(features, cnty)
        tmp = tmp[~tmp.is_empty]

        # Buffer setback
        tmp.loc[:, 'geometry'] = tmp.buffer(setback)

        setbacks = [(geom, 1) for geom in tmp['geometry']]

        return setbacks

    @staticmethod
    def _get_feature_paths(features_path):
        """
        Ensure features path is valid

        Parameters
        ----------
        features_path : str
            Path to features file

        Returns
        -------
        features_path : list
            Features path as a list
        """
        if not os.path.exists(features_path):
            msg = '{} is not a valid file path!'.format(features_path)
            logger.error(msg)
            raise FileNotFoundError(msg)

        return [features_path]

    def _parse_regs(self, regs_fpath):
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
        regs = super()._parse_regs(regs_fpath)

        mask = regs['Feature Type'] == 'Transmission'
        regs = regs.loc[mask]

        return regs

    @classmethod
    def run(cls, excl_h5, features_fpath, out_dir, hub_height,
            rotor_diameter, regs_fpath=None, multiplier=None,
            chunks=(128, 128), max_workers=None, replace=False, hsds=False):
        """
        Compute setbacks from given features and write them to a geotiff.
        If a wind regulations file is given compute local setbacks, otherwise
        compute generic setbacks using the given multiplier and the turbine
        tip-height.
        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing exclusion layers, will also be the
            location of any new setback layers
        features_fpath : str
            Path to shape file with transmission or rail features to compute
            setbacks from
        out_dir : str
            Directory to save geotiff containing rasterized setbacks into
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
        replace : bool, optional
            Flag to replace geotiff if it already exists, by default False
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        """
        geotiff = os.path.basename(features_fpath).split('.')[0]
        geotiff += '.geotiff'
        geotiff = os.path.join(out_dir, geotiff)
        if os.path.exists(geotiff) and not replace:
            msg = ('{} already exists, setbacks will not be re-computed '
                   'unless replace=True'.format(geotiff))
            logger.error(msg)
        else:
            setbacks = cls(excl_h5, hub_height, rotor_diameter,
                           regs_fpath=regs_fpath, multiplier=multiplier,
                           hsds=hsds, chunks=chunks)

            logger.info("Computing setbacks from {} and saving "
                        "to {}".format(features_fpath, geotiff))
            setbacks.compute_setbacks(features_fpath, geotiff=geotiff,
                                      max_workers=max_workers,
                                      replace=replace)


class RailWindSetbacks(TransmissionWindSetbacks):
    """
    Rail Wind setbacks, computed against a single set of railroad features,
    instead of state level features, uses the same approach as
    TransmissionWindSetbacks
    """
    def _parse_regs(self, regs_fpath):
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
        # pylint: disable=bad-super-call
        regs = super(TransmissionWindSetbacks, self)._parse_regs(regs_fpath)

        mask = regs['Feature Type'] == 'Railroads'
        regs = regs.loc[mask]

        return regs
