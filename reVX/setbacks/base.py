# -*- coding: utf-8 -*-
"""
Compute setbacks exclusions
"""
from abc import ABC, abstractmethod
from concurrent.futures import as_completed
from warnings import warn
import os
import logging
import pathlib
import numpy as np
import geopandas as gpd
from rasterio import features
from shapely.geometry import shape

from rex.utilities import parse_table, SpawnProcessPool, log_mem
from reV.handlers.exclusions import ExclusionLayers
from reVX.utilities.exclusions_converter import ExclusionsConverter
from reVX.utilities.utilities import log_versions

logger = logging.getLogger(__name__)


class BaseSetbacks(ABC):
    """
    Create exclusions layers for setbacks
    """

    def __init__(self, excl_fpath, base_setback_dist, regulations_fpath=None,
                 multiplier=None, hsds=False, chunks=(128, 128)):
        """
        Parameters
        ----------
        excl_fpath : str
            Path to .h5 file containing exclusion layers, will also be
            the location of any new setback layers
        base_setback_dist : float | int
            Base setback distance (m). This value will be used to
            calculate the setback distance when a multiplier is provided
            either via the `regulations_fpath`csv or the `multiplier`
            input. In this case, the setbacks will be calculated using
            `base_setback_dist * multiplier`.
        regulations_fpath : str | None, optional
            Path to regulations .csv file. At a minimum, this csv must
            contain the following columns: `Value Type`, which
            specifies wether the value is a multiplier or static height,
            `Value`, which specifies the numeric value of the setback or
            multiplier, and `FIPS`, which specifies a unique 5-digit
            code for each county (this can be an integer - no leading
            zeros required). Typically, this csv will also have a
            `Feature Type` column that labels the type of setback
            that each row represents. Valid options for the `Value Type`
            are:
                - "Max-tip Height Multiplier"
                - "Rotor-Diameter Multiplier"
                - "Hub-height Multiplier"
                - "Structure Height multiplier"
                - "Meters"
            If this input is `None`, a generic setback of
            `base_setback_dist * multiplier` is used. By default `None`.
        multiplier : int | float | None, optional
            A setback multiplier to use if regulations are not supplied.
            This multiplier will be applied to the plant height. If
            supplied along with `regulations_fpath`, this input will be
            ignored. By default `None`.
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on
            AWS behind HSDS. By default `False`.
        chunks : tuple, optional
            Chunk size to use for setback layers, if None use default
            chunk size in excl_fpath. By default `(128, 128)`.
        """
        log_versions(logger)
        self._base_setback_dist = base_setback_dist
        self._excl_fpath = excl_fpath
        self._hsds = hsds
        excl_props = self._parse_excl_properties(excl_fpath, chunks, hsds=hsds)
        self._shape, self._chunks, self._profile = excl_props

        self._regulations, self._multi = self._preflight_check(
            regulations_fpath, multiplier
        )

    def __repr__(self):
        msg = "{} for {}".format(self.__class__.__name__, self._excl_fpath)
        return msg

    @staticmethod
    def _parse_excl_properties(excl_fpath, chunks, hsds=False):
        """Parse shape, chunk size, and profile from exclusions file.

        Parameters
        ----------
        excl_fpath : str
            Path to .h5 file containing exclusion layers, will also be
            the location of any new setback layers
        chunks : tuple | None
            Chunk size of exclusions datasets
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on
            AWS behind HSDS. By default `False`.

        Returns
        -------
        shape : tuple
            Shape of exclusions datasets
        chunks : tuple | None
            Chunk size of exclusions datasets
        profile : str
            GeoTiff profile for exclusions datasets
        """
        with ExclusionLayers(excl_fpath, hsds=hsds) as exc:
            dset_shape = exc.shape
            profile = exc.profile
            if chunks is None:
                chunks = exc.chunks

        if len(chunks) < 3:
            chunks = (1, ) + chunks

        if len(dset_shape) < 3:
            dset_shape = (1, ) + dset_shape

        logger.debug('Exclusions properties:\n'
                     'shape : {}\n'
                     'chunks : {}\n'
                     'profile : {}\n'
                     .format(dset_shape, chunks, profile))

        return dset_shape, chunks, profile

    def _preflight_check(self, regulations_fpath, multiplier):
        """Apply preflight checks to the regulations path and multiplier.

        Run preflight checks on setback inputs:
        1) Ensure either a regulations .csv is provided, or
           a setback multiplier
        2) Ensure regulations has county FIPS, map regulations to county
           geometries from exclusions .h5 file

        Parameters
        ----------
        regulations_fpath : str | None
            Path to regulations .csv file, if `None`, create global
            setbacks.
        multiplier : int | float | str | None
            Setback multiplier to use if regulations are not supplied.

        Returns
        -------
        regulations: `geopandas.GeoDataFrame` | None
            GeoDataFrame with county level setback regulations merged
            with county geometries, use for intersecting with setback
            features.
        Multiplier : float | None
            Generic setbacks multiplier
        """
        if regulations_fpath:
            if multiplier:
                msg = ('A regulation .csv file was also provided and '
                       'will be used to determine setback multipliers!')
                logger.warning(msg)
                warn(msg)

            multiplier = None
            regulations = self._parse_regulations(regulations_fpath)
            logger.debug('Computing setbacks using regulations provided in: {}'
                         .format(regulations_fpath))
        elif multiplier:
            regulations = None
            logger.debug('Computing setbacks using generic plant height '
                         'multiplier of {}'.format(multiplier))
        else:
            msg = ('Computing setbacks requires either a regulations '
                   '.csv file or a generic multiplier!')
            logger.error(msg)
            raise RuntimeError(msg)

        return regulations, multiplier

    def _parse_regulations(self, regulations_fpath):
        """Parse regulations file.

        Parameters
        ----------
        regulations_fpath : str
            Path to regulations .csv file.

        Returns
        -------
        regulations: `geopandas.GeoDataFrame`
            GeoDataFrame with county level setback regulations merged
            with county geometries, use for intersecting with setback
            features.
        """
        try:
            regulations = parse_table(regulations_fpath)
            regulations = self._parse_county_regulations(regulations)
            log_mem(logger)

            out_path = regulations_fpath.split('.')[0] + '.gpkg'
            logger.debug('Saving regulations with county geometries as: '
                         '{}'.format(out_path))
            regulations.to_file(out_path, driver='GPKG')
        except ValueError:
            regulations = gpd.read_file(regulations_fpath)

        fips_check = regulations['geometry'].isnull()
        if fips_check.any():
            msg = ('The following county FIPS were requested in the '
                   'regulations but were not available in the '
                   'Exclusions "cnty_fips" layer:\n{}'
                   .format(regulations.loc[fips_check, 'FIPS']))
            logger.error(msg)
            raise RuntimeError(msg)

        return regulations.to_crs(crs=self.crs)

    def _parse_county_regulations(self, regulations):
        """Parse the county regulations.

        Parse regulations, combine with county geometries from
        exclusions .h5 file. The county geometries are intersected with
        features to compute county specific setbacks.

        Parameters
        ----------
        regulations : pandas.DataFrame
            Regulations table

        Returns
        -------
        regulations: `geopandas.GeoDataFrame`
            GeoDataFrame with county level setback regulations merged
            with county geometries, use for intersecting with setback
            features.
        """
        if 'FIPS' not in regulations:
            msg = ('Regulations does not have county FIPS! Please add a '
                   '"FIPS" columns with the unique county FIPS values.')
            logger.error(msg)
            raise RuntimeError(msg)

        if 'geometry' not in regulations:
            regulations['geometry'] = None

        regulations = regulations.set_index('FIPS')

        logger.info('Merging county geometries w/ local regulations')
        with ExclusionLayers(self._excl_fpath) as exc:
            fips = exc['cnty_fips']
            profile = exc.get_layer_profile('cnty_fips')

        s = features.shapes(
            fips.astype(np.int32),
            transform=profile['transform']
        )
        for p, v in s:
            v = int(v)
            if v in regulations.index:
                regulations.at[v, 'geometry'] = shape(p)

        regulations = gpd.GeoDataFrame(
            regulations, crs=self.crs, geometry='geometry'
        )

        return regulations.reset_index()

    @property
    def base_setback_dist(self):
        """The base setback distance, in meters.

        Returns
        -------
        float
        """
        return self._base_setback_dist

    @property
    def generic_setback(self):
        """Default setback of plant height * multiplier.

        This value is used for global setbacks.

        Returns
        -------
        float
        """
        if self.multiplier is None:
            setback = None
        else:
            setback = self.base_setback_dist * self.multiplier

        return setback

    @property
    def multiplier(self):
        """Generic setback multiplier.

        Returns
        -------
        int | float
        """
        return self._multi

    @property
    def arr_shape(self):
        """Rasterize array shape.

        Returns
        -------
        tuple
        """
        return self._shape

    @property
    def profile(self):
        """Geotiff profile.

        Returns
        -------
        dict
        """
        return self._profile

    @property
    def crs(self):
        """Coordinate reference system.

        Returns
        -------
        str
        """
        return self.profile['crs']

    @property
    def regulations(self):
        """Regulations table.

        Returns
        -------
        geopandas.GeoDataFrame | None
        """
        return self._regulations

    def _parse_features(self, features_fpath):
        """Abstract method to parse features.

        Parameters
        ----------
        features_fpath : str
            Path to file containing features to setback from.

        Returns
        -------
        `geopandas.GeoDataFrame`
            Geometries of features to setback from in exclusion
            coordinate system.
        """
        return gpd.read_file(features_fpath).to_crs(crs=self.crs)

    # pylint: disable=unused-argument
    def _check_regulations(self, features_fpath):
        """Reduce regulations to state corresponding to features_fpath.

        Parameters
        ----------
        features_fpath : str
            Path to shape file with features to compute setbacks from.

        Returns
        -------
        regulations : geopandas.GeoDataFrame | None
            Regulations table.
        """
        logger.debug('Computing setbacks for regulations in {} counties'
                     .format(len(self.regulations)))

        return self.regulations

    @abstractmethod
    def get_regulation_setback(self, county_regulations):
        """Compute the setback distance for the county.

        Compute the setback distance (in meters) from the
        county regulations or the plant height.

        Parameters
        ----------
        county_regulations : pandas.Series
            Pandas Series with regulations for a single county
            or feature type. At a minimum, this Series must
            contain the following columns: `Value Type`, which
            specifies wether the value is a multiplier or static height,
            `Value`, which specifies the numeric value of the setback or
            multiplier. Valid options for the `Value Type` are:
                - "Max-tip Height Multiplier"
                - "Rotor-Diameter Multiplier"
                - "Hub-height Multiplier"
                - "Structure Height multiplier"
                - "Meters"

        Returns
        -------
        setback : float | None
            Setback distance in meters, or `None` if the setback
            `Value Type` was not recognized.
        """

    @staticmethod
    def _compute_local_setbacks(features, cnty, setback):
        """Compute local features setbacks.

        This method will compute the setbacks using a county-specific
        regulations file that specifies either a static setback or a
        multiplier value that will be used along with plant height
        specifications to compute the setback.

        Parameters
        ----------
        features : geopandas.GeoDataFrame
            Features to setback from.
        cnty : geopandas.GeoDataFrame
            Regulations for a single county.
        setback : int
            Setback distance in meters.

        Returns
        -------
        setbacks : list
            List of setback geometries.
        """
        logger.debug('- Computing setbacks for county FIPS {}'
                     .format(cnty.iloc[0]['FIPS']))
        log_mem(logger)
        mask = features.centroid.within(cnty['geometry'].values[0])
        tmp = features.loc[mask]
        tmp.loc[:, 'geometry'] = tmp.buffer(setback)

        setbacks = [(geom, 1) for geom in tmp['geometry']]

        return setbacks

    def _no_exclusions_array(self):
        """Get an array of the correct shape reprenting no exclusions.

        The array contains all zeros, and a new one is created
        for every function call.

        Returns
        -------
        np.array
            Array of zeros representing no exclusions.
        """
        return np.zeros(self.arr_shape[1:], dtype='uint8')

    def _rasterize_setbacks(self, shapes):
        """Convert setbacks geometries into exclusions array.

        Parameters
        ----------
        shapes : list, optional
            List of (geometry, 1) pairs to rasterize. Each geometry is a
            feature buffered by the desired setback distance in meters.
            If `None` or empty list, returns array of zeros.

        Returns
        -------
        arr : ndarray
            Rasterized array of setbacks.
        """
        logger.debug('Generating setbacks exclusion array of shape {}'
                     .format(self.arr_shape))
        log_mem(logger)
        arr = self._no_exclusions_array()
        if shapes:
            features.rasterize(shapes=shapes,
                               out=arr,
                               out_shape=self.arr_shape[1:],
                               fill=0,
                               transform=self.profile['transform'])

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
            exists in the exclusion .h5 file. By default `False`.
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
        """Compute local setbacks for all counties either.

        Parameters
        ----------
        features_fpath : str
            Path to shape file with features to compute setbacks from
        max_workers : int, optional
            Number of workers to use for setback computation, if 1 run
            in serial, if > 1 run in parallel with that many workers,
            if `None` run in parallel on all available cores.
            By default `None`.

        Returns
        -------
        setbacks : ndarray
            Raster array of setbacks.
        """
        regulations = self._check_regulations(features_fpath)
        if regulations.empty:
            return self._no_exclusions_array()

        setbacks = []
        setback_features = self._parse_features(features_fpath)
        if max_workers is None:
            max_workers = os.cpu_count()

        log_mem(logger)
        if max_workers > 1:
            logger.info('Computing local setbacks in parallel using {} '
                        'workers'.format(max_workers))
            loggers = [__name__, 'reVX']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                futures = []
                for i in range(len(regulations)):
                    cnty = regulations.iloc[[i]].copy()
                    setback = self.get_regulation_setback(cnty.iloc[0])
                    if setback is not None:
                        idx = setback_features.sindex.intersection(
                            cnty.total_bounds
                        )
                        cnty_feats = setback_features.iloc[list(idx)].copy()
                        future = exe.submit(self._compute_local_setbacks,
                                            cnty_feats, cnty, setback)
                        futures.append(future)

                for i, future in enumerate(as_completed(futures)):
                    setbacks.extend(future.result())
                    logger.debug('Computed setbacks for {} of {} counties'
                                 .format((i + 1), len(regulations)))
        else:
            logger.info('Computing local setbacks in serial')
            for i in range(len(regulations)):
                cnty = regulations.iloc[[i]]
                setback = self.get_regulation_setback(cnty.iloc[0])
                if setback is not None:
                    idx = setback_features.sindex.intersection(
                        cnty.total_bounds
                    )
                    cnty_feats = setback_features.iloc[list(idx)]
                    setbacks.extend(self._compute_local_setbacks(cnty_feats,
                                                                 cnty,
                                                                 setback))
                    logger.debug('Computed setbacks for {} of {} counties'
                                 .format((i + 1), len(regulations)))

        return self._rasterize_setbacks(setbacks)

    def compute_generic_setbacks(self, features_fpath):
        """Compute generic setbacks.

        This method will compute the setbacks using a generic setback
        of `base_setback_dist * multiplier`.

        Parameters
        ----------
        features_fpath : str
            Path to shape file with features to compute setbacks from.

        Returns
        -------
        setbacks : ndarray
            Raster array of setbacks
        """
        logger.info('Computing generic setbacks')
        setback_features = self._parse_features(features_fpath)
        setback_features.loc[:, 'geometry'] = setback_features.buffer(
            self.generic_setback
        )
        setbacks = [(geom, 1) for geom in setback_features['geometry']]

        return self._rasterize_setbacks(setbacks)

    def compute_setbacks(self, features_fpath, max_workers=None,
                         geotiff=None, replace=False):
        """
        Compute setbacks for all states either in serial or parallel.
        Existing setbacks are computed if a regulations file was
        supplied during class initialization, otherwise generic setbacks
        are computed.

        Parameters
        ----------
        features_fpath : str
            Path to shape file with features to compute setbacks from
        max_workers : int, optional
            Number of workers to use for setback computation, if 1 run
            in serial, if > 1 run in parallel with that many workers,
            if `None`, run in parallel on all available cores.
            By default `None`.
        geotiff : str, optional
            Path to save geotiff containing rasterized setbacks.
            By default `None`.
        replace : bool, optional
            Flag to replace geotiff if it already exists.
            By default `False`.

        Returns
        -------
        setbacks : ndarray
            Raster array of setbacks
        """
        if self._regulations is not None:
            setbacks = self.compute_local_setbacks(features_fpath,
                                                   max_workers=max_workers)
        else:
            setbacks = self.compute_generic_setbacks(features_fpath)

        if geotiff is not None:
            logger.debug('Writing setbacks to {}'.format(geotiff))
            self._write_setbacks(geotiff, setbacks, replace=replace)

        return setbacks

    @staticmethod
    def _get_feature_paths(features_fpath):
        """Ensure features path exists and return as list.

        Parameters
        ----------
        features_fpath : str
            Path to features file. This path can contain
            any pattern that can be used in the glob function.
            For example, `/path/to/features/[A]*` would match
            with all the features in the direcotry
            `/path/to/features/` that start with "A". This input
            can also be a directory, but that directory must ONLY
            contain feature files. If your feature files are mixed
            with other files or directories, use something like
            `/path/to/features/*.geojson`.

        Returns
        -------
        features_fpath : list
            Features path as a list of strings.

        Notes
        -----
        This method is required for `run` classmethods for
        feature setbacks that are spread out over multiple
        files.
        """
        glob_path = pathlib.Path(features_fpath)
        if glob_path.is_dir():
            glob_path = glob_path / '*'

        paths = [str(f) for f in glob_path.parent.glob(glob_path.name)]
        if not paths:
            msg = 'No files found matching the input {!r}!'
            msg = msg.format(features_fpath)
            logger.error(msg)
            raise FileNotFoundError(msg)

        return paths
