# -*- coding: utf-8 -*-
"""
Compute setbacks exclusions
"""
from concurrent.futures import as_completed
from warnings import warn
from itertools import product
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


def features_with_centroid_in_county(features, cnty):
    """Find features with centroids within the given county.

    Parameters
    ----------
    features : geopandas.GeoDataFrame
        Features to setback from.
    cnty : geopandas.GeoDataFrame
        Regulations for a single county.

    Returns
    -------
    features : geopandas.GeoDataFrame
        Features that have centroid in county.
    """

    mask = features.centroid.within(cnty['geometry'].values[0])
    return features.loc[mask]


def features_clipped_to_county(features, cnty):
    """Clip features to the given county geometry.

    Parameters
    ----------
    features : geopandas.GeoDataFrame
        Features to setback from.
    cnty : geopandas.GeoDataFrame
        Regulations for a single county.

    Returns
    -------
    features : geopandas.GeoDataFrame
        Features clipped to county geometry.
    """
    tmp = gpd.clip(features, cnty)
    return tmp[~tmp.is_empty]


class BaseSetbacks:
    """
    Create exclusions layers for setbacks
    """

    def __init__(self, excl_fpath, base_setback_dist, regulations_fpath=None,
                 multiplier=None, hsds=False, chunks=(128, 128),
                 weights_calculation_upscale_factor=None):
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
                - "Structure Height Multiplier"
                - "Meters"
            If this input is `None`, a generic setback of
            `base_setback_dist * multiplier` is used. By default `None`.
        multiplier : int | float | str | None, optional
            A setback multiplier to use if regulations are not supplied.
            This multiplier will be applied to the ``base_setback_dist``
            to calculate the setback. If supplied along with
            ``regulations_fpath``, this input will be used to apply a
            setback to all counties not listed in the regulations file.
            By default `None`.
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on
            AWS behind HSDS. By default `False`.
        chunks : tuple, optional
            Chunk size to use for setback layers, if None use default
            chunk size in excl_fpath. By default `(128, 128)`.
        weights_calculation_upscale_factor : int, optional
            If this value is an int > 1, the output will be a layer with
            **inclusion** weight values instead of exclusion booleans.
            For example, a cell that was previously excluded with a
            a boolean mask (value of 1) may instead be converted to an
            inclusion weight value of 0.75, meaning that 75% of the area
            corresponding to that point should be included (i.e. the
            exclusion feature only intersected a small portion - 25% -
            of the cell). This percentage inclusion value is calculated
            by upscaling the output array using this input value,
            rasterizing the exclusion features onto it, and counting the
            number of resulting sub-cells excluded by the feature. For
            example, setting the value to `3` would split each output
            cell into nine sub-cells - 3 divisions in each dimension.
            After the feature is rasterized on this high-resolution
            sub-grid, the area of the non-excluded sub-cells is totaled
            and divided by the area of the original cell to obtain the
            final inclusion percentage. Therefore, a larger upscale
            factor results in more accurate percentage values. However,
            this process is memory intensive and scales quadratically
            with the upscale factor. A good way to estimate your minimum
            memory requirement is to use the following formula:

            .. math:: memory (GB) = s_0 * s_1 * (sf^2 * 2 + 4) / 1073741824,

            where :math:`s_0` and :math:`s_1` are the dimensions (shape)
            of your exclusion layer and :math:`sf` is the scale factor
            (be sure to add several GB for any other overhead required
            by the rest of the process). If `None` (or a value <= 1),
            this process is skipped and the output is a boolean
            exclusion mask. By default `None`.
        """
        log_versions(logger)
        self._base_setback_dist = base_setback_dist
        self._excl_fpath = excl_fpath
        self._hsds = hsds
        excl_props = self._parse_excl_properties(excl_fpath, chunks, hsds=hsds)
        self._shape, self._chunks, self._profile = excl_props
        self._scale_factor = weights_calculation_upscale_factor or 1
        self._scale_factor = int(self._scale_factor // 1)
        self._regulations = None
        self._multi = multiplier

        self._preflight_check(regulations_fpath)

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

    def _preflight_check(self, regulations_fpath):
        """Apply preflight checks to the regulations path and multiplier.

        Run preflight checks on setback inputs:
        1) Ensure either a regulations .csv or
           a setback multiplier (or both) is provided
        2) Ensure regulations has county FIPS, map regulations to county
           geometries from exclusions .h5 file

        Parameters
        ----------
        regulations_fpath : str | None
            Path to regulations .csv file, if `None`, create global
            setbacks.
        """
        if regulations_fpath:
            self._regulations = self._parse_regulations(regulations_fpath)
            logger.debug('Computing setbacks using regulations provided in: {}'
                         .format(regulations_fpath))

        if self._multi:
            logger.debug('Computing setbacks using base setback distance '
                         'multiplier of {}'.format(self._multi))

        if not regulations_fpath and not self._multi:
            msg = ('Computing setbacks requires either a regulations '
                   '.csv file or a generic multiplier!')
            logger.error(msg)
            raise RuntimeError(msg)

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
        except ValueError:
            regulations = gpd.read_file(regulations_fpath)

        log_mem(logger)
        fips_check = regulations['geometry'].isnull()
        if fips_check.any():
            msg = ('The following county FIPS were requested in the '
                   'regulations but were not available in the '
                   'Exclusions "cnty_fips" layer:\n{}'
                   .format(regulations.loc[fips_check, 'FIPS']))
            logger.error(msg)
            raise RuntimeError(msg)

        new_col_names = {col: col.lower().title()
                         for col in regulations.columns
                         if col.lower() not in {"geometry", "fips"}}
        regulations = regulations.rename(new_col_names, axis=1)

        missing = [col for col in ["Feature Type", "Value Type", "Value"]
                   if col not in regulations]
        if any(missing):
            msg = ('Regulations are missing the following required columns: {}'
                   .format(missing))
            logger.error(msg)
            raise RuntimeError(msg)

        for col in ["Feature Type", "Value Type", "Value", "FIPS"]:
            regulations = regulations[~regulations[col].isna()]

        feature_types = regulations['Feature Type'].str.strip().str.lower()
        regulations['Feature Type'] = feature_types

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

        regulations = regulations[~regulations['FIPS'].isna()]
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
        """Default setback of base setback distance * multiplier.

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

    def get_regulation_setback(self, county_regulations):
        """Compute the setback distance for the county.

        Compute the setback distance (in meters) from the
        county regulations or the base setback distance.

        Parameters
        ----------
        county_regulations : pandas.Series
            Pandas Series with regulations for a single county
            or feature type. At a minimum, this Series must
            contain the following columns: `Value Type`, which
            specifies wether the value is a multiplier or static height,
            `Value`, which specifies the numeric value of the setback or
            multiplier. Valid options for the `Value Type` are:
                - "Structure Height Multiplier"
                - "Meters"

        Returns
        -------
        setback : float | None
            Setback distance in meters, or `None` if the setback
            `Value Type` was not recognized.
        """

        setback_type = county_regulations["Value Type"].strip()
        setback = float(county_regulations["Value"])
        if setback_type.lower() == "structure height multiplier":
            setback *= self.base_setback_dist
        elif setback_type.lower() != "meters":
            msg = ("Cannot create setback for {}, expecting "
                   '"Structure Height Multiplier", or '
                   '"Meters", but got {}'
                   .format(county_regulations["County"], setback_type))
            logger.warning(msg)
            warn(msg)
            setback = None

        return setback

    def _compute_local_setbacks(self, features, cnty, setback):
        """Compute local features setbacks.

        This method will compute the setbacks using a county-specific
        regulations file that specifies either a static setback or a
        multiplier value that will be used along with the base setback
        distance to compute the setback.

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
        features = self._feature_filter(features, cnty)
        features.loc[:, 'geometry'] = features.buffer(setback)

        setbacks = [(geom, 1) for geom in features['geometry']]

        return setbacks

    @staticmethod
    def _feature_filter(features, cnty):
        """Filter the features given a county."""
        return features_with_centroid_in_county(features, cnty)

    def _no_exclusions_array(self, multiplier=1):
        """Get an array of the correct shape representing no exclusions.

        The array contains all zeros, and a new one is created
        for every function call.

        Parameters
        ----------
        multiplier : int, optional
            Integer multiplier value used to scale up the dimensions of
            the array exclusions array (e.g. multiplier of 3 turns an
            array of shape (10, 20) into an array of shape (30, 60)).

        Returns
        -------
        np.array
            Array of zeros representing no exclusions.
        """
        high_res_shape = tuple(x * multiplier for x in self.arr_shape[1:])
        return np.zeros(high_res_shape, dtype='uint8')

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

        if self._scale_factor > 1:
            return self._rasterize_to_weights(shapes)

        return self._rasterize_to_mask(shapes)

    def _rasterize_to_mask(self, shapes):
        """Rasterize features with to an exclusion mask."""

        arr = self._no_exclusions_array()
        if shapes:
            features.rasterize(shapes=shapes,
                               out=arr,
                               out_shape=arr.shape[1:],
                               fill=0,
                               transform=self.profile['transform'])

        return arr

    def _rasterize_to_weights(self, shapes):
        """Rasterize features to weights using a high-resolution array."""

        if not shapes:
            return 1 - self._no_exclusions_array()

        hr_arr = self._no_exclusions_array(multiplier=self._scale_factor)
        new_transform = list(self.profile['transform'])[:6]
        new_transform[0] = new_transform[0] / self._scale_factor
        new_transform[4] = new_transform[4] / self._scale_factor

        features.rasterize(shapes=shapes,
                           out=hr_arr,
                           out_shape=hr_arr.shape[1:],
                           fill=0,
                           transform=new_transform)

        arr = self._aggregate_high_res(hr_arr)
        return 1 - (arr / self._scale_factor ** 2)

    def _aggregate_high_res(self, hr_arr):
        """Aggregate the high resolution exclusions array to output shape. """

        arr = self._no_exclusions_array().astype(np.float32)
        for i, j in product(range(self._scale_factor),
                            range(self._scale_factor)):
            arr += hr_arr[i::self._scale_factor, j::self._scale_factor]
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
            return self._rasterize_setbacks(shapes=None)

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
        setbacks = self._compute_merged_setbacks(features_fpath,
                                                 max_workers=max_workers)

        if geotiff is not None:
            logger.debug('Writing setbacks to {}'.format(geotiff))
            self._write_setbacks(geotiff, setbacks, replace=replace)

        return setbacks

    def _compute_merged_setbacks(self, features_fpath, max_workers=None):
        """Compute and merge local and generic setbacks, if necessary. """

        if self.generic_setback is not None and self.regulations is None:
            return self.compute_generic_setbacks(features_fpath)

        if self.regulations is not None and self.generic_setback is None:
            return self.compute_local_setbacks(features_fpath,
                                               max_workers=max_workers)

        generic_setbacks = self.compute_generic_setbacks(features_fpath)
        local_setbacks = self.compute_local_setbacks(features_fpath,
                                                     max_workers=max_workers)
        return self._merge_setbacks(generic_setbacks, local_setbacks,
                                    features_fpath)

    def _merge_setbacks(self, generic_setbacks, local_setbacks,
                        features_fpath):
        """Merge local setbacks onto the generic setbacks."""
        logger.info('Merging local setbacks onto the generic setbacks')

        regulations = self._check_regulations(features_fpath)
        with ExclusionLayers(self._excl_fpath) as exc:
            fips = exc['cnty_fips']

        local_setbacks_mask = np.isin(fips, regulations["FIPS"].unique())

        generic_setbacks[local_setbacks_mask] = (
            local_setbacks[local_setbacks_mask]
        )
        return generic_setbacks

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

    @classmethod
    def run(cls, excl_fpath, features_path, out_dir, base_setback_dist,
            regulations_fpath=None, multiplier=None,
            chunks=(128, 128), weights_calculation_upscale_factor=None,
            max_workers=None, replace=False, hsds=False):
        """
        Compute setbacks and write them to a geotiff. If a regulations
        file is given, compute local setbacks, otherwise compute generic
        setbacks using the given multiplier and the base setback
        distance. If both are provided, generic and local setbacks are
        merged such that the local setbacks override the generic ones.

        Parameters
        ----------
        excl_fpath : str
            Path to .h5 file containing exclusion layers, will also be
            the location of any new setback layers.
        features_path : str
            Path to file or directory feature shape files.
            This path can contain any pattern that can be used in the
            glob function. For example, `/path/to/features/[A]*` would
            match with all the features in the directory
            `/path/to/features/` that start with "A". This input
            can also be a directory, but that directory must ONLY
            contain feature files. If your feature files are mixed
            with other files or directories, use something like
            `/path/to/features/*.geojson`.
        out_dir : str
            Directory to save setbacks geotiff(s) into
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
                - "Structure Height Multiplier"
                - "Meters"
            If this input is `None`, a generic setback of
            `base_setback_dist * multiplier` is used. By default `None`.
        multiplier : int | float | str | None, optional
            A setback multiplier to use if regulations are not supplied.
            This multiplier will be applied to the ``base_setback_dist``
            to calculate the setback. If supplied along with
            ``regulations_fpath``, this input will be used to apply a
            setback to all counties not listed in the regulations file.
            By default `None`.
        chunks : tuple, optional
            Chunk size to use for setback layers, if None use default
            chunk size in excl_fpath, By default `(128, 128)`.
        weights_calculation_upscale_factor : int, optional
            If this value is an int > 1, the output will be a layer with
            **inclusion** weight values instead of exclusion booleans.
            For example, a cell that was previously excluded with a
            a boolean mask (value of 1) may instead be converted to an
            inclusion weight value of 0.75, meaning that 75% of the area
            corresponding to that point should be included (i.e. the
            exclusion feature only intersected a small portion - 25% -
            of the cell). This percentage inclusion value is calculated
            by upscaling the output array using this input value,
            rasterizing the exclusion features onto it, and counting the
            number of resulting sub-cells excluded by the feature. For
            example, setting the value to `3` would split each output
            cell into nine sub-cells - 3 divisions in each dimension.
            After the feature is rasterized on this high-resolution
            sub-grid, the area of the non-excluded sub-cells is totaled
            and divided by the area of the original cell to obtain the
            final inclusion percentage. Therefore, a larger upscale
            factor results in more accurate percentage values. However,
            this process is memory intensive and scales quadratically
            with the upscale factor. A good way to estimate your minimum
            memory requirement is to use the following formula:

            .. math:: memory (GB) = s_0 * s_1 * ((sf^2) * 2 + 4) / 1073741824,

            where :math:`s_0` and :math:`s_1` are the dimensions (shape)
            of your exclusion layer and :math:`sf` is the scale factor
            (be sure to add several GB for any other overhead required
            by the rest of the process). If `None` (or a value <= 1),
            this process is skipped and the output is a boolean
            exclusion mask. By default `None`.
        max_workers : int, optional
            Number of workers to use for setback computation, if 1 run
            in serial, if > 1 run in parallel with that many workers,
            if `None`, run in parallel on all available cores.
            By default `None`.
        replace : bool, optional
            Flag to replace geotiff if it already exists.
            By default `False`.
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on
            AWS behind HSDS. By default `False`.
        """
        scale_factor = weights_calculation_upscale_factor
        setbacks = cls(excl_fpath, base_setback_dist=base_setback_dist,
                       regulations_fpath=regulations_fpath,
                       multiplier=multiplier,
                       hsds=hsds, chunks=chunks,
                       weights_calculation_upscale_factor=scale_factor)

        features_path = setbacks._get_feature_paths(features_path)
        for fpath in features_path:
            geotiff = os.path.basename(fpath)
            geotiff = ".".join(geotiff.split('.')[:-1] + ['tif'])
            geotiff = os.path.join(out_dir, geotiff)

            if os.path.exists(geotiff) and not replace:
                msg = ('{} already exists, setbacks will not be re-computed '
                       'unless replace=True'.format(geotiff))
                logger.error(msg)
            else:
                logger.info("Computing setbacks from {} and saving "
                            "to {}".format(fpath, geotiff))
                setbacks.compute_setbacks(fpath, geotiff=geotiff,
                                          max_workers=max_workers,
                                          replace=replace)
