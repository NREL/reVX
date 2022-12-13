# -*- coding: utf-8 -*-
"""
Driver class to compute exclusions
"""
import h5py
import json
import os
import logging
from abc import ABC, abstractmethod
from concurrent.futures import as_completed
from warnings import warn

import numpy as np
import geopandas as gpd
from pyproj.crs import CRS
import rasterio
from rasterio import features as rio_features
from shapely.geometry import shape

from rex import Outputs
from rex.utilities import SpawnProcessPool, log_mem
from reV.handlers.exclusions import ExclusionLayers
from reVX.handlers.geotiff import Geotiff
from reVX.utilities.utilities import log_versions
from reVX.utilities.exceptions import ExclusionsCheckError

logger = logging.getLogger(__name__)


class AbstractExclusionCalculatorInterface(ABC):
    """Abstract Exclusion Calculator Interface. """

    @property
    @abstractmethod
    def no_exclusions_array(self):
        """np.array: Array representing no exclusions. """
        raise NotImplementedError

    @abstractmethod
    def pre_process_regulations(self):
        """Reduce regulations to correct state and features.

        When implementing this method, make sure to update
        `self._regulations.df`.
        """
        raise NotImplementedError

    @abstractmethod
    def parse_features(self):
        """Parse features used to compute exclusions.

        Warnings
        --------
        Use caution when calling this method, especially in multiple
        processes, as the returned feature files may be quite large.
        Reading 100 GB feature files in each of 36 sub-processes will
        quickly overwhelm your RAM.
        """
        raise NotImplementedError

    @abstractmethod
    def _local_exclusions_arguments(self, regulation_value, cnty):
        """Compile and return arguments to `compute_local_exclusions`.

        This method should return a list or tuple of extra args to be
        passed to `compute_local_exclusions`. Do not include the
        `regulation_value` or `cnty`.

        Parameters
        ----------
        regulation_value : float | int
            Regulation value for county.
        cnty : geopandas.GeoDataFrame
            Regulations for a single county.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def compute_local_exclusions(regulation_value, cnty, *args):
        """Compute local feature exclusions.

        This method should compute the exclusions using the information
        about the input county.

        Parameters
        ----------
        regulation_value : float | int
            Regulation value for county.
        cnty : geopandas.GeoDataFrame
            Regulations for a single county.
        *args
            Other arguments required for local exclusion calculation.

        Returns
        -------
        exclusions : np.ndarray
            Array of exclusions.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_generic_exclusions(self, max_workers=None):
        """Compute generic exclusions.

        This method should compute the exclusions using a generic
        regulation value (`self._regulations.generic`).

        Parameters
        ----------
        max_workers : int, optional
            Number of workers to use for exclusions computation, if 1
            run in serial, if > 1 run in parallel with that many
            workers, if `None` run in parallel on all available cores.
            By default `None`.

        Returns
        -------
        exclusions : ndarray
            Raster array of exclusions
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def input_output_filenames(cls, out_dir, features_fpath, kwargs):
        """Generate pairs of input/output file names.

        Parameters
        ----------
        out_dir : str
            Path to output file directory.
        features_fpath : str
            Path to features file. This path can contain
            any pattern that can be used in the glob function.
            For example, `/path/to/features/[A]*` would match
            with all the features in the directory
            `/path/to/features/` that start with "A". This input
            can also be a directory, but that directory must ONLY
            contain feature files. If your feature files are mixed
            with other files or directories, use something like
            `/path/to/features/*.geojson`.
        kwargs : dict
            Dictionary of extra keyword-argument pairs used to
            instantiate the `exclusion_class`.

        Yields
        ------
        tuple
            An input-output filename pair.
        """
        raise NotImplementedError


class AbstractBaseExclusionsMerger(AbstractExclusionCalculatorInterface):
    """
    Create exclusions layers for exclusions
    """

    def __init__(self, excl_fpath, regulations, features, hsds=False):
        """
        Parameters
        ----------
        excl_fpath : str
            Path to .h5 file containing exclusion layers, will also be
            the location of any new exclusion layers
        regulations : `~reVX.utilities.AbstractBaseRegulations` subclass
            A regulations object used to extract exclusion regulation
            values.
        features : str
            Path to file containing features to compute exclusions from.
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on
            AWS behind HSDS. By default `False`.
        """
        log_versions(logger)
        self._excl_fpath = excl_fpath
        self._regulations = regulations
        self._features = features
        self._hsds = hsds
        self._fips = self._profile = None
        self._set_profile()
        self._process_regulations(regulations.df)
        self.features = self.parse_features()

    def __repr__(self):
        msg = "{} for {}".format(self.__class__.__name__, self._excl_fpath)
        return msg

    def _set_profile(self):
        """Extract profile from excl h5."""
        with ExclusionLayers(self._excl_fpath, hsds=self._hsds) as f:
            self._profile = f.profile

    def _process_regulations(self, regulations_df):
        """Parse the county regulations.

        Parse regulations, combine with county geometries from
        exclusions .h5 file. The county geometries are intersected with
        features to compute county specific exclusions.

        Parameters
        ----------
        regulations : pandas.DataFrame
            Regulations table

        Returns
        -------
        regulations: `geopandas.GeoDataFrame`
            GeoDataFrame with county level exclusion regulations merged
            with county geometries, use for intersecting with exclusion
            features.
        """
        if regulations_df is None:
            return

        with ExclusionLayers(self._excl_fpath, hsds=self._hsds) as exc:
            self._fips = exc['cnty_fips']
            cnty_fips_profile = exc.get_layer_profile('cnty_fips')

        if 'FIPS' not in regulations_df:
            msg = ('Regulations does not have county FIPS! Please add a '
                   '"FIPS" columns with the unique county FIPS values.')
            logger.error(msg)
            raise RuntimeError(msg)

        if 'geometry' not in regulations_df:
            regulations_df['geometry'] = None

        regulations_df = regulations_df[~regulations_df['FIPS'].isna()]
        regulations_df = regulations_df.set_index('FIPS')

        logger.info('Merging county geometries w/ local regulations')
        s = rio_features.shapes(
            self._fips.astype(np.int32),
            transform=cnty_fips_profile['transform']
        )
        for p, v in s:
            v = int(v)
            if v in regulations_df.index:
                regulations_df.at[v, 'geometry'] = shape(p)

        regulations_df = gpd.GeoDataFrame(
            regulations_df,
            crs=self.profile['crs'],
            geometry='geometry'
        )
        regulations_df = regulations_df.reset_index()
        regulations_df = regulations_df.to_crs(crs=self.profile['crs'])
        self._regulations.df = regulations_df

    @property
    def profile(self):
        """dict: Geotiff profile. """
        return self._profile

    @property
    def regulations_table(self):
        """Regulations table.

        Returns
        -------
        geopandas.GeoDataFrame | None
        """
        return self._regulations.df

    @regulations_table.setter
    def regulations_table(self, regulations_table):
        self._process_regulations(regulations_table)

    def _write_exclusions(self, geotiff, exclusions, replace=False):
        """
        Write exclusions to geotiff, replace if requested

        Parameters
        ----------
        geotiff : str
            Path to geotiff file to save exclusions too
        exclusions : ndarray
            Rasterized array of exclusions.
        replace : bool, optional
            Flag to replace local layer data with arr if file already
            exists on disk. By default `False`.
        """
        if os.path.exists(geotiff):
            _error_or_warn(geotiff, replace)

        ExclusionsConverter.write_geotiff(geotiff, self.profile, exclusions)

    def _write_layer(self, out_layer, exclusions, replace=False):
        """Write exclusions to H5, replace if requested

        Parameters
        ----------
        out_layer : str
            Name of new exclusion layer to add to h5.
        exclusions : ndarray
            Rasterized array of exclusions.
        replace : bool, optional
            Flag to replace local layer data with arr if layer already
            exists in the exclusion .h5 file. By default `False`.
        """
        with ExclusionLayers(self._excl_fpath, hsds=self._hsds) as exc:
            layers = exc.layers

        if out_layer in layers:
            _error_or_warn(out_layer, replace)

        try:
            description = self.description
        except AttributeError:
            description = None

        ExclusionsConverter._write_layer(self._excl_fpath, out_layer,
                                         self.profile, exclusions,
                                         description=description)

    def compute_all_local_exclusions(self, max_workers=None):
        """Compute local exclusions for all counties either.

        Parameters
        ----------
        max_workers : int, optional
            Number of workers to use for exclusions computation, if 1
            run in serial, if > 1 run in parallel with that many
            workers, if `None` run in parallel on all available cores.
            By default `None`.

        Returns
        -------
        exclusions : ndarray
            Raster array of exclusions.
        """
        max_workers = max_workers or os.cpu_count()

        log_mem(logger)
        if max_workers > 1:
            logger.info('Computing local exclusions in parallel using {} '
                        'workers'.format(max_workers))
            spp_kwargs = {"max_workers": max_workers,
                          "loggers": [__name__, 'reVX']}
            with SpawnProcessPool(**spp_kwargs) as exe:
                exclusions = self._compute_exclusions_spp(exe, max_workers)
        else:
            logger.info('Computing local exclusions in serial')
            exclusions = None
            for i, (exclusion, cnty) in enumerate(self._regulations):
                args = self._local_exclusions_arguments(exclusion, cnty)
                local_exclusions = self.compute_local_exclusions(exclusion,
                                                                 cnty,
                                                                 *args)
                exclusions = self._combine_exclusions(exclusions,
                                                      local_exclusions,
                                                      cnty['FIPS'].unique())
                logger.debug('Computed exclusions for {} of {} counties'
                             .format((i + 1), len(self.regulations_table)))

        return exclusions

    def _compute_exclusions_spp(self, exe, max_submissions):
        """Compute exclusions in parallel using futures. """
        futures, exclusions = {}, None

        for ind, regulation_info in enumerate(self._regulations, start=1):
            exclusion_value, cnty = regulation_info
            logger.info('Computing exclusions for {}/{} counties'
                        .format(ind, len(self.regulations_table)))
            args = self._local_exclusions_arguments(exclusion_value, cnty)
            future = exe.submit(self.compute_local_exclusions,
                                exclusion_value, cnty, *args)
            futures[future] = cnty['FIPS'].unique()
            if ind % max_submissions == 0:
                exclusions = self._collect_futures(futures, exclusions)
        exclusions = self._collect_futures(futures, exclusions)
        return exclusions

    def _collect_futures(self, futures, exclusions):
        """Collect all futures from the input dictionary. """
        for future in as_completed(futures):
            exclusions = self._combine_exclusions(exclusions,
                                                  future.result(),
                                                  futures.pop(future))
            log_mem(logger)
        return exclusions

    def compute_exclusions(self, out_layer=None, out_tiff=None, replace=False,
                           max_workers=None):
        """
        Compute exclusions for all states either in serial or parallel.
        Existing exclusions are computed if a regulations file was
        supplied during class initialization, otherwise generic exclusions
        are computed.

        Parameters
        ----------
        out_layer : str, optional
            Name to save rasterized exclusions under in .h5 file.
            If `None`, exclusions will not be written to the .h5 file.
            By default `None`.
        out_tiff : str, optional
            Path to save geotiff containing rasterized exclusions.
            If `None`, exclusions will not be written to a geotiff file.
            By default `None`.
        replace : bool, optional
            Flag to replace geotiff if it already exists.
            By default `False`.
        max_workers : int, optional
            Number of workers to use for exclusion computation, if 1 run
            in serial, if > 1 run in parallel with that many workers,
            if `None`, run in parallel on all available cores.
            By default `None`.

        Returns
        -------
        exclusions : ndarray
            Raster array of exclusions
        """
        exclusions = self._compute_merged_exclusions(max_workers=max_workers)

        if out_layer is not None:
            logger.info('Saving exclusion layer to {} as {}'
                        .format(self._excl_fpath, out_layer))
            self._write_layer(out_layer, exclusions, replace=replace)

        if out_tiff is not None:
            logger.debug('Writing exclusions to {}'.format(out_tiff))
            self._write_exclusions(out_tiff, exclusions, replace=replace)

        return exclusions

    def _compute_merged_exclusions(self, max_workers=None):
        """Compute and merge local and generic exclusions, if necessary. """
        mw = max_workers

        if self._regulations.locals_exist:
            self.pre_process_regulations()

        generic_exclusions_exist = self._regulations.generic_exists
        local_exclusions_exist = self._regulations.locals_exist

        if not generic_exclusions_exist and not local_exclusions_exist:
            msg = ("Found no exclusions to compute: No regulations detected, "
                   "and generic multiplier not set.")
            logger.error(msg)
            raise ValueError(msg)

        if generic_exclusions_exist and not local_exclusions_exist:
            return self.compute_generic_exclusions(max_workers=mw)

        if local_exclusions_exist and not generic_exclusions_exist:
            local_excl = self.compute_all_local_exclusions(max_workers=mw)
            nea = self.no_exclusions_array.astype(local_excl.dtype)
            return self._merge_exclusions(nea, local_excl)

        generic_exclusions = self.compute_generic_exclusions(max_workers=mw)
        local_exclusions = self.compute_all_local_exclusions(max_workers=mw)
        return self._merge_exclusions(generic_exclusions, local_exclusions)

    def _merge_exclusions(self, generic_exclusions, local_exclusions):
        """Merge local exclusions onto the generic exclusions."""
        logger.info('Merging local exclusions onto the generic exclusions')

        self.pre_process_regulations()
        local_fips = self.regulations_table["FIPS"].unique()
        return self._combine_exclusions(generic_exclusions, local_exclusions,
                                        local_fips)

    def _combine_exclusions(self, existing, additional, cnty_fips):
        """Combine local exclusions using FIPS code"""
        if existing is None:
            return additional

        local_exclusions_mask = np.isin(self._fips, cnty_fips)
        existing[local_exclusions_mask] = additional[local_exclusions_mask]
        return existing

    @classmethod
    def run(cls, excl_fpath, features_path, out_dir, regulations,
            max_workers=None, replace=False, out_layers=None, hsds=False,
            **kwargs):
        """
        Compute exclusions and write them to a geotiff. If a regulations
        file is given, compute local exclusions, otherwise compute
        generic exclusions. If both are provided, generic and local
        exclusions are merged such that the local exclusions override
        the generic ones.

        Parameters
        ----------
        excl_fpath : str
            Path to .h5 file containing exclusion layers, will also be
            the location of any new exclusion layers.
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
            Directory to save exclusion geotiff(s) into
        regulations : `~reVX.utilities.AbstractBaseRegulations` subclass
            A regulations object used to extract exclusion regulation
            distances.
        max_workers : int, optional
            Number of workers to use for exclusion computation, if 1 run
            in serial, if > 1 run in parallel with that many workers,
            if `None`, run in parallel on all available cores.
            By default `None`.
        replace : bool, optional
            Flag to replace geotiff if it already exists.
            By default `False`.
        out_layers : dict, optional
            Dictionary mapping feature file names (with extension) to
            names of layers under which exclusions should be saved in
            the `excl_fpath` .h5 file. If `None` or empty dictionary,
            no layers are saved to the h5 file. By default `None`.
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on
            AWS behind HSDS. By default `False`.
        **kwargs
            Keyword args to exclusions calculator class.
        """

        out_layers = out_layers or {}
        cls_init_kwargs = {"excl_fpath": excl_fpath,
                           "regulations": regulations}
        cls_init_kwargs.update(kwargs)
        files = cls.input_output_filenames(out_dir, features_path,
                                           cls_init_kwargs)
        for f_in, f_out in files:
            if os.path.exists(f_out) and not replace:
                msg = ('{} already exists, exclusions will not be re-computed '
                       'unless replace=True'.format(f_out))
                logger.error(msg)
            else:
                logger.info("Computing exclusions from {} and saving "
                            "to {}".format(f_in, f_out))
                out_layer = out_layers.get(os.path.basename(f_in))
                exclusions = cls(excl_fpath=excl_fpath,
                                 regulations=regulations, features=f_in,
                                 hsds=hsds, **kwargs)
                exclusions.compute_exclusions(out_tiff=f_out,
                                              out_layer=out_layer,
                                              max_workers=max_workers,
                                              replace=replace)


class ExclusionsConverter:
    """
    Convert exclusion layers between .h5 and .tif (geotiff)
    """
    def __init__(self, excl_h5, hsds=False, chunks=(128, 128), replace=True):
        """
        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing or to contain exclusion layers
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        chunks : tuple, optional
            Chunk size of exclusions in .h5 and Geotiffs, by default (128, 128)
        replace : bool, optional
            Flag to replace existing layers if needed, by default True
        """
        log_versions(logger)
        self._excl_h5 = excl_h5
        self._hsds = hsds
        self._chunks = chunks
        self._replace = replace

    def __repr__(self):
        msg = "{} for {}".format(self.__class__.__name__, self._excl_h5)
        return msg

    def __getitem__(self, layer):
        """
        Parameters
        ----------
        layer : str
            Layer to extract data for

        Returns
        -------
        profile : dict
            Geotiff profile (attributes)
        values : ndarray
            Geotiff data
        """

        if layer not in self.layers:
            msg = "{} is not present in {}".format(layer, self._excl_h5)
            logger.error(msg)
            raise KeyError(msg)

        profile, values = self._extract_layer(self._excl_h5, layer,
                                              hsds=self._hsds)
        return profile, values

    def __setitem__(self, layer, geotiff):
        """
        Parameters
        ----------
        layer : str
            Layer to set
        geotiff : str
            Path to GeoTiff to load data from
        """
        self.geotiff_to_layer(layer, geotiff)

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
    def _init_h5(excl_h5, geotiff, chunks=(128, 128)):
        """
        Initialize exclusions .h5 file from geotiff:
        - Transfer profile, shape, and meta

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing exclusion layers
        geotiff : str
            Path to geotiff file
        chunks : tuple
            Chunk size of exclusions in Geotiff
        """
        logger.debug('\t- Initializing {} from {}'
                     .format(excl_h5, geotiff))
        with Geotiff(geotiff, chunks=chunks) as src:
            profile = src.profile
            shape = src.shape
            lat, lon = src.lat_lon
            logger.debug('\t- "profile", "meta", and "shape" extracted from {}'
                         .format(geotiff))

        try:
            with h5py.File(excl_h5, mode='w') as dst:
                dst.attrs['profile'] = json.dumps(profile)
                logger.debug('\t- Default profile:\n{}'.format(profile))
                dst.attrs['shape'] = shape
                logger.debug('\t- Default shape:\n{}'.format(shape))
                dst.attrs['chunks'] = chunks
                logger.debug('\t- Default chunks:\n{}'.format(chunks))

                dst.create_dataset('latitude', shape=lat.shape,
                                   dtype=np.float32, data=lat,
                                   chunks=chunks)
                logger.debug('\t- latitude coordiantes created')

                dst.create_dataset('longitude', shape=lon.shape,
                                   dtype=np.float32, data=lon,
                                   chunks=chunks)
                logger.debug('\t- longitude coordiantes created')
        except Exception:
            logger.exception("Error initilizing {}".format(excl_h5))
            if os.path.exists(excl_h5):
                os.remove(excl_h5)

    @staticmethod
    def _check_crs(baseline_crs, test_crs, ignore_keys=('no_defs',)):
        """
        Compare baseline and test crs values

        Parameters
        ----------
        baseline_crs : dict
            Baseline CRS to use a truth, must be a dict
        test_crs : dict
            Test CRS to compare with baseline, must be a dictionary
        ignore_keys : tuple
            Keys to not check

        Returns
        -------
        bad_crs : bool
            Flag if crs' do not match
        """
        bad_crs = False
        for k, true_v in baseline_crs.items():
            if k not in ignore_keys:
                test_v = test_crs.get(k, true_v)
                if true_v != test_v:
                    bad_crs = True

        return bad_crs

    @classmethod
    def _check_geotiff(cls, excl_h5, geotiff, chunks=(128, 128),
                       transform_atol=0.01, coord_atol=0.001):
        """
        Compare geotiff with exclusion layer, raise any errors

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing exclusion layers
        geotiff : str
            Path to geotiff file
        chunks : tuple
            Chunk size of exclusions in Geotiff
        transform_atol : float
            Absolute tolerance parameter when comparing geotiff transform data.
        coord_atol : float
            Absolute tolerance parameter when comparing new un-projected
            geotiff coordinates against previous coordinates.
        """
        with Geotiff(geotiff, chunks=chunks) as tif:
            with ExclusionLayers(excl_h5) as h5:
                if tif.bands > 1:
                    error = ('{} contains more than one band!'
                             .format(geotiff))
                    logger.error(error)
                    raise ExclusionsCheckError(error)

                if not np.array_equal(h5.shape, tif.shape):
                    error = ('Shape of exclusion data in {} and {} do not '
                             'match!'.format(geotiff, excl_h5))
                    logger.error(error)
                    raise ExclusionsCheckError(error)

                profile = h5.profile
                h5_crs = CRS.from_string(profile['crs']).to_dict()
                tif_crs = CRS.from_string(tif.profile['crs']).to_dict()
                bad_crs = cls._check_crs(h5_crs, tif_crs)
                if bad_crs:
                    error = ('Geospatial "crs" in {} and {} do not match!'
                             '\n {} !=\n {}'
                             .format(geotiff, excl_h5, tif_crs, h5_crs))
                    logger.error(error)
                    raise ExclusionsCheckError(error)

                if not np.allclose(profile['transform'],
                                   tif.profile['transform'],
                                   atol=transform_atol):
                    error = ('Geospatial "transform" in {} and {} do not '
                             'match!\n {} !=\n {}'
                             .format(geotiff, excl_h5, profile['transform'],
                                     tif.profile['transform']))
                    logger.error(error)
                    raise ExclusionsCheckError(error)

                lat, lon = tif.lat_lon
                if not np.allclose(h5.latitude, lat, atol=coord_atol):
                    error = ('Latitude coordinates {} and {} do not match to '
                             'within {} degrees!'
                             .format(geotiff, excl_h5, coord_atol))
                    logger.error(error)
                    raise ExclusionsCheckError(error)

                if not np.allclose(h5.longitude, lon, atol=coord_atol):
                    error = ('Longitude coordinates {} and {} do not match to '
                             'within {} degrees!'
                             .format(geotiff, excl_h5, coord_atol))
                    logger.error(error)
                    raise ExclusionsCheckError(error)

    @classmethod
    def parse_tiff(cls, geotiff, excl_h5=None, chunks=(128, 128),
                   check_tiff=True, transform_atol=0.01, coord_atol=0.001):
        """
        Extract exclusion layer from given geotiff, compare with excl_h5
        if provided

        Parameters
        ----------
        geotiff : str
            Path to geotiff file
        excl_h5 : str, optional
            Path to .h5 file containing exclusion layers, by default None
        chunks : tuple, optional
            Chunk size of exclusions in Geotiff, by default (128, 128)
        check_tiff : bool, optional
            Flag to check tiff profile and coordinates against exclusion .h5
            profile and coordinates, by default True
        transform_atol : float, optional
            Absolute tolerance parameter when comparing geotiff transform data,
            by default 0.01
        coord_atol : float, optional
            Absolute tolerance parameter when comparing new un-projected
            geotiff coordinates against previous coordinates, by default 0.001

        Returns
        -------
        profile : dict
            Geotiff profile (attributes)
        values : ndarray
            Geotiff data
        """
        if excl_h5 is not None and check_tiff:
            cls._check_geotiff(excl_h5, geotiff, chunks=chunks,
                               transform_atol=transform_atol,
                               coord_atol=coord_atol)

        with Geotiff(geotiff, chunks=chunks) as tif:
            profile, values = tif.profile, tif.values

        return profile, values

    @staticmethod
    def _write_layer(excl_h5, layer, profile, values, chunks=(128, 128),
                     description=None, scale_factor=None):
        """
        Write exclusion layer to .h5 file

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing exclusion layers
        layer : str
            Dataset name in .h5 file
        profile : dict
            Geotiff profile (attributes)
        values : ndarray
            Geotiff data
        chunks : tuple
            Chunk size of dataset in .h5 file
        description : str
            Description of exclusion layer
        scale_factor : int | float, optional
            Scale factor to use to scale geotiff data when added to the .h5
            file, by default None
        """
        if len(chunks) < 3:
            chunks = (1, ) + chunks

        if values.ndim < 3:
            values = np.expand_dims(values, 0)

        with h5py.File(excl_h5, mode='a') as f:
            if layer in f:
                ds = f[layer]
                ds[...] = values
                logger.debug('\t- {} values replaced'.format(layer))
            else:
                ds = f.create_dataset(layer, shape=values.shape,
                                      dtype=values.dtype, chunks=chunks,
                                      data=values)
                logger.debug('\t- {} created and loaded'.format(layer))

            ds.attrs['profile'] = json.dumps(profile)
            logger.debug('\t- Unique profile for {} added:\n{}'
                         .format(layer, profile))
            if description is not None:
                ds.attrs['description'] = description
                logger.debug('\t- Description for {} added:\n{}'
                             .format(layer, description))

            if scale_factor is not None:
                ds.attrs['scale_factor'] = scale_factor
                logger.debug('\t- scale_factor for {} added:\n{}'
                             .format(layer, scale_factor))

    @classmethod
    def _geotiff_to_h5(cls, excl_h5, layer, geotiff, chunks=(128, 128),
                       check_tiff=True, transform_atol=0.01, coord_atol=0.001,
                       description=None, scale_factor=None, dtype='int16'):
        """
        Transfer geotiff exclusions to h5 confirming they match existing layers

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing exclusion layers
        layer : str
            Layer to extract
        geotiff : str
            Path to geotiff file
        chunks : tuple, optional
            Chunk size of exclusions in Geotiff, by default (128, 128)
        check_tiff : bool, optional
            Flag to check tiff profile and coordinates against exclusion .h5
            profile and coordinates, by default True
        transform_atol : float, optional
            Absolute tolerance parameter when comparing geotiff transform data,
            by default 0.01
        coord_atol : float, optional
            Absolute tolerance parameter when comparing new un-projected
            geotiff coordinates against previous coordinates, by default 0.001
        description : str, optional
            Description of exclusion layer, by default None
        scale_factor : int | float, optional
            Scale factor to use to scale geotiff data when added to the .h5
            file, by default None
        dtype : str, optional
            Dtype to save geotiff data as in the .h5 file. Only used when
            'scale_factor' is not None, by default 'int16'
        """
        logger.debug('\t- {} being extracted from {} and added to {}'
                     .format(layer, geotiff, os.path.basename(excl_h5)))

        profile, values = cls.parse_tiff(
            geotiff, excl_h5=excl_h5, chunks=chunks, check_tiff=check_tiff,
            transform_atol=transform_atol, coord_atol=coord_atol)

        if scale_factor is not None:
            attrs = {'scale_factor': scale_factor}
            values = Outputs._check_data_dtype(layer, values, dtype,
                                               attrs=attrs)

        cls._write_layer(excl_h5, layer, profile, values,
                         chunks=chunks, description=description,
                         scale_factor=scale_factor)

    @staticmethod
    def write_geotiff(geotiff, profile, values):
        """
        Write values to geotiff with given profile

        Parameters
        ----------
        geotiff : str
            Path to geotiff file to save data to
        profile : dict
            Geotiff profile (attributes)
        values : ndarray
            Geotiff data
        """
        out_dir = os.path.dirname(geotiff)
        if not os.path.exists(out_dir):
            logger.debug("Creating {}".format(out_dir))
            os.makedirs(out_dir)

        if values.shape[0] != 1:
            values = np.expand_dims(values, 0)

        dtype = values.dtype.name
        profile['dtype'] = dtype
        if np.issubdtype(dtype, np.integer):
            dtype_max = np.iinfo(dtype).max
        else:
            dtype_max = np.finfo(dtype).max

        profile['nodata'] = dtype_max

        with rasterio.open(geotiff, 'w', **profile) as f:
            f.write(values)
            logger.debug('\t- {} created'.format(geotiff))

    @classmethod
    def _extract_layer(cls, excl_h5, layer, geotiff=None, hsds=False):
        """
        Extract given layer from exclusions .h5 file and write to geotiff .tif

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing exclusion layers
        layer : str
            Layer to extract
        geotiff : str
            Path to geotiff file
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS

        Returns
        -------
        profile : dict
            Geotiff profile (attributes)
        values : ndarray
            Geotiff data
        """
        logger.debug('\t - Extracting {} from {}'
                     .format(layer, os.path.basename(excl_h5)))
        with ExclusionLayers(excl_h5, hsds=hsds) as f:
            profile = f.get_layer_profile(layer)
            values = f.get_layer_values(layer)

        if geotiff is not None:
            logger.debug('\t- Writing {} to {}'.format(layer, geotiff))
            cls.write_geotiff(geotiff, profile, values)

        return profile, values

    def geotiff_to_layer(self, layer, geotiff, check_tiff=True,
                         transform_atol=0.01, coord_atol=0.001,
                         description=None, scale_factor=None, dtype='int16'):
        """
        Transfer geotiff exclusions to h5 confirming they match existing layers

        Parameters
        ----------
        layer : str
            Layer to extract
        geotiff : str
            Path to geotiff file
        check_tiff : bool, optional
            Flag to check tiff profile and coordinates against exclusion .h5
            profile and coordinates, by default True
        transform_atol : float, optional
            Absolute tolerance parameter when comparing geotiff transform data,
            by default 0.01
        coord_atol : float, optional
            Absolute tolerance parameter when comparing new un-projected
            geotiff coordinates against previous coordinates, by default 0.001
        description : str, optional
            Description of exclusion layer, by default None
        scale_factor : int | float, optional
            Scale factor to use to scale geotiff data when added to the .h5
            file, by default None
        dtype : str, optional
            Dtype to save geotiff data as in the .h5 file. Only used when
            'scale_factor' is not None, by default 'int16'
        """
        if not os.path.exists(self._excl_h5):
            self._init_h5(self._excl_h5, geotiff, chunks=self._chunks)

        if layer in self.layers:
            msg = ("{} is already present in {}"
                   .format(layer, self._excl_h5))
            if self._replace:
                msg += " and will be replaced"
                logger.warning(msg)
                warn(msg)
            else:
                msg += ", to 'replace' set to True"
                logger.error(msg)
                raise KeyError(msg)

        self._geotiff_to_h5(self._excl_h5, layer, geotiff,
                            chunks=self._chunks,
                            check_tiff=check_tiff,
                            transform_atol=transform_atol,
                            coord_atol=coord_atol,
                            description=description,
                            scale_factor=scale_factor,
                            dtype=dtype)

    def layer_to_geotiff(self, layer, geotiff):
        """
        Extract desired layer from .h5 file and write to geotiff .tif

        Parameters
        ----------
        layer : str
            Layer to extract
        geotiff : str
            Path to geotiff file
        """
        self._extract_layer(self._excl_h5, layer, geotiff=geotiff,
                            hsds=self._hsds)

    @classmethod
    def layers_to_h5(cls, excl_h5, layers, chunks=(128, 128),
                     replace=True, check_tiff=True,
                     transform_atol=0.01, coord_atol=0.001,
                     descriptions=None, scale_factors=None):
        """
        Create exclusions .h5 file, or load layers into existing exclusion .h5
        file from provided geotiffs

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing or to contain exclusion layers
        layers : list | dict
            List of geotiffs to load
            or dictionary mapping goetiffs to the layers to load
        chunks : tuple, optional
            Chunk size of exclusions in Geotiff, by default (128, 128)
        replace : bool, optional
            Flag to replace existing layers if needed, by default True
        check_tiff : bool, optional
            Flag to check tiff profile and coordinates against exclusion .h5
            profile and coordinates, by default True
        transform_atol : float, optional
            Absolute tolerance parameter when comparing geotiff transform data,
            by default 0.01
        coord_atol : float, optional
            Absolute tolerance parameter when comparing new un-projected
            geotiff coordinates against previous coordinates, by default 0.001
        description : dict, optional
            Description of exclusion layers, by default None
        scale_factor : dict, optional
            Scale factors and dtypes to use when scaling given layers,
            by default None
        """
        if isinstance(layers, list):
            layers = {os.path.basename(lyr).split('.')[0]: lyr
                      for lyr in layers}

        if descriptions is None:
            descriptions = {}

        if scale_factors is None:
            scale_factors = {}

        excls = cls(excl_h5, chunks=chunks, replace=replace)
        logger.info('Creating {}'.format(excl_h5))
        for layer, geotiff in layers.items():
            logger.info('- Transfering {}'.format(layer))
            description = descriptions.get(layer, None)
            scale = scale_factors.get(layer, None)
            if scale is not None:
                scale_factor = scale['scale_factor']
                dtype = scale['dtype']
            else:
                scale_factor = None
                dtype = None

            excls.geotiff_to_layer(layer, geotiff, check_tiff=check_tiff,
                                   transform_atol=transform_atol,
                                   coord_atol=coord_atol,
                                   description=description,
                                   scale_factor=scale_factor,
                                   dtype=dtype)

    @classmethod
    def extract_layers(cls, excl_h5, layers, chunks=(128, 128),
                       hsds=False):
        """
        Extract given layers from exclusions .h5 file and save to disk
        as GeoTiffs

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing or to contain exclusion layers
        layers : dict
            Dictionary mapping layers to geotiffs to create
        chunks : tuple
            Chunk size of exclusions in .h5 and Geotiffs
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        excls = cls(excl_h5, chunks=chunks, hsds=hsds)
        logger.info('Extracting layers from {}'.format(excl_h5))
        for layer, geotiff in layers.items():
            logger.info('- Extracting {}'.format(geotiff))
            excls.layer_to_geotiff(layer, geotiff)

    @classmethod
    def extract_all_layers(cls, excl_h5, out_dir, chunks=(128, 128),
                           hsds=False):
        """
        Extract all layers from exclusions .h5 file and save to disk
        as GeoTiffs

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing or to contain exclusion layers
        out_dir : str
            Path to output directory into which layers should be saved as
            GeoTiffs
        chunks : tuple
            Chunk size of exclusions in .h5 and Geotiffs
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        excls = cls(excl_h5, chunks=chunks, hsds=hsds)
        logger.info('Extracting layers from {}'.format(excl_h5))
        for layer in excls.layers:
            geotiff = os.path.join(out_dir, "{}.tif".format(layer))
            logger.info('- Extracting {}'.format(geotiff))
            excls.layer_to_geotiff(layer, geotiff)


def _error_or_warn(name, replace):
    """If replace, throw warning, otherwise throw error. """
    if not replace:
        msg = ('{} already exists. To replace it set "replace=True"'
               .format(name))
        logger.error(msg)
        raise IOError(msg)
    else:
        msg = ('{} already exists and will be replaced!'
               .format(name))
        logger.warning(msg)
        warn(msg)
