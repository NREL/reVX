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
import pandas as pd
from pyproj.crs import CRS
import rasterio
from rasterio import features as rio_features
from shapely.geometry import shape

from rex import Outputs
from rex.utilities import SpawnProcessPool, log_mem
from reV.handlers.exclusions import ExclusionLayers
from reVX.handlers.geotiff import Geotiff
from reVX.handlers.layered_h5 import LayeredH5
from reVX.utilities.utilities import log_versions

logger = logging.getLogger(__name__)


class AbstractExclusionCalculatorInterface(ABC):
    """Abstract Exclusion Calculator Interface. """

    @property
    @abstractmethod
    def no_exclusions_array(self):
        """np.array: Array representing no exclusions. """
        raise NotImplementedError

    @property
    @abstractmethod
    def exclusion_merge_func(self):
        """callable: Function to merge overlapping exclusion layers. """
        raise NotImplementedError

    @abstractmethod
    def pre_process_regulations(self):
        """Reduce regulations to correct state and features.

        When implementing this method, make sure to update
        `self._regulations.df`.
        """
        raise NotImplementedError

    @abstractmethod
    def _local_exclusions_arguments(self, regulation_value, county):
        """Compile and yield arguments to `compute_local_exclusions`.

        This method should yield lists or tuples of extra args to be
        passed to `compute_local_exclusions`. Do not include the
        `regulation_value` or `county`.

        Parameters
        ----------
        regulation_value : float | int
            Regulation value for county.
        county : geopandas.GeoDataFrame
            Regulations for a single county.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def compute_local_exclusions(regulation_value, county, *args):
        """Compute local feature exclusions.

        This method should compute the exclusions using the information
        about the input county.

        Parameters
        ----------
        regulation_value : float | int
            Regulation value for county.
        county : geopandas.GeoDataFrame
            Regulations for a single county.
        *args
            Other arguments required for local exclusion calculation.

        Returns
        -------
        exclusions : np.ndarray
            Array of exclusions.
        slices : 2-tuple of `slice`
            X and Y slice objects defining where in the original array
            the exclusion data should go.
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
        shapes_from_raster = rio_features.shapes(
            self._fips.astype(np.int32),
            transform=cnty_fips_profile['transform']
        )
        county_regs = []
        for polygon, fips_code in shapes_from_raster:
            fips_code = int(fips_code)
            if fips_code in regulations_df.index:
                local_regs = regulations_df.loc[[fips_code]].copy()
                local_regs['geometry'] = shape(polygon)
                county_regs.append(local_regs)

        if county_regs:
            regulations_df = pd.concat(county_regs)

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

        Geotiff.write(geotiff, self.profile, exclusions)

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

        LayeredH5(self._excl_fpath).write_layer_to_h5(exclusions, out_layer,
                                                      self.profile,
                                                      description=description)

    def _county_exclusions(self):
        """Yield county exclusion arguments. """
        for ind, regulation_info in enumerate(self._regulations, start=1):
            exclusion, cnty = regulation_info
            logger.debug('Computing exclusions for {}/{} counties'
                         .format(ind, len(self.regulations_table)))
            for args in self._local_exclusions_arguments(exclusion, cnty):
                yield exclusion, cnty, args

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
        mw = max_workers or os.cpu_count()

        log_mem(logger)
        exclusions = None
        if mw > 1:
            logger.info('Computing local exclusions in parallel using {} '
                        'workers'.format(mw))
            spp_kwargs = {"max_workers": mw, "loggers": [__name__, 'reVX']}
            with SpawnProcessPool(**spp_kwargs) as exe:
                exclusions = self._compute_local_exclusions_in_chunks(exe, mw)

        else:
            logger.info('Computing local exclusions in serial')
            for ind, cnty_inf in enumerate(self._county_exclusions(), start=1):
                exclusion_value, cnty, args = cnty_inf
                out = self.compute_local_exclusions(exclusion_value, cnty,
                                                    *args)
                local_exclusions, slices = out
                fips = cnty['FIPS'].unique()
                exclusions = self._combine_exclusions(exclusions,
                                                      local_exclusions,
                                                      fips, slices)
                logger.debug("Computed exclusions for {:,} counties"
                             .format(ind))
        if exclusions is None:
            exclusions = self.no_exclusions_array

        return exclusions

    def _compute_local_exclusions_in_chunks(self, exe, max_submissions):
        """Compute exclusions in parallel using futures. """
        futures, exclusions = {}, None

        for ind, reg in enumerate(self._county_exclusions(), start=1):
            exclusion_value, cnty, args = reg
            future = exe.submit(self.compute_local_exclusions,
                                exclusion_value, cnty, *args)
            futures[future] = cnty['FIPS'].unique()
            if ind % max_submissions == 0:
                exclusions = self._collect_local_futures(futures, exclusions)
        exclusions = self._collect_local_futures(futures, exclusions)
        return exclusions

    def _collect_local_futures(self, futures, exclusions):
        """Collect all futures from the input dictionary. """
        for future in as_completed(futures):
            new_exclusions, slices = future.result()
            exclusions = self._combine_exclusions(exclusions,
                                                  new_exclusions,
                                                  futures.pop(future),
                                                  slices=slices)
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
            logger.warning(msg)
            warn(msg)
            return self.no_exclusions_array

        if generic_exclusions_exist and not local_exclusions_exist:
            return self.compute_generic_exclusions(max_workers=mw)

        if local_exclusions_exist and not generic_exclusions_exist:
            local_excl = self.compute_all_local_exclusions(max_workers=mw)
            # merge ensures local exclusions are clipped county boundaries
            return self._merge_exclusions(None, local_excl)

        generic_exclusions = self.compute_generic_exclusions(max_workers=mw)
        local_exclusions = self.compute_all_local_exclusions(max_workers=mw)
        return self._merge_exclusions(generic_exclusions, local_exclusions)

    def _merge_exclusions(self, generic_exclusions, local_exclusions):
        """Merge local exclusions onto the generic exclusions."""
        logger.info('Merging local exclusions onto the generic exclusions')

        local_fips = self.regulations_table["FIPS"].unique()
        return self._combine_exclusions(generic_exclusions, local_exclusions,
                                        local_fips, replace_existing=True)

    def _combine_exclusions(self, existing, additional=None, cnty_fips=None,
                            slices=None, replace_existing=False):
        """Combine local exclusions using FIPS code"""
        if additional is None:
            return existing

        if existing is None:
            existing = self.no_exclusions_array.astype(additional.dtype)

        if slices is None:
            slices = tuple([slice(None)] * len(existing.shape))

        if cnty_fips is None:
            local_exclusions = slice(None)
        else:
            local_exclusions = np.isin(self._fips[slices], cnty_fips)

        if replace_existing:
            new_local_exclusions = additional[local_exclusions]
        else:
            new_local_exclusions = self.exclusion_merge_func(
                existing[slices][local_exclusions],
                additional[local_exclusions])
        existing[slices][local_exclusions] = new_local_exclusions
        return existing

    @classmethod
    def run(cls, excl_fpath, features_path, out_fn, regulations,
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
        out_fn : str
            Path to output geotiff where exclusion data should be
            stored.
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

        if os.path.exists(out_fn) and not replace:
            msg = ('{} already exists, exclusions will not be re-computed '
                   'unless replace=True'.format(out_fn))
            logger.error(msg)
        else:
            logger.info("Computing exclusions from {} and saving "
                        "to {}".format(features_path, out_fn))
            out_layer = out_layers.get(os.path.basename(features_path))
            exclusions = cls(excl_fpath=excl_fpath, regulations=regulations,
                             features=features_path, hsds=hsds, **kwargs)
            exclusions.compute_exclusions(out_tiff=out_fn, out_layer=out_layer,
                                          max_workers=max_workers,
                                          replace=replace)


def _error_or_warn(name, replace):
    """If replace, throw warning, otherwise throw error. """
    if not replace:
        msg = ('{} already exists. To replace it set "replace=True"'
               .format(name))
        logger.error(msg)
        raise IOError(msg)

    msg = ('{} already exists and will be replaced!'.format(name))
    logger.warning(msg)
    warn(msg)
