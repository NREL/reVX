# -*- coding: utf-8 -*-
"""
Compute setbacks exclusions
"""
import logging
import os
import numpy as np
import geopandas as gpd

from rex.utilities import log_mem

from reVX.setbacks.base import AbstractBaseSetbacks


logger = logging.getLogger(__name__)


class ParcelSetbacks(AbstractBaseSetbacks):
    """Parcel setbacks - facilitates the use of negative buffers. """

    @staticmethod
    def compute_local_exclusions(regulation_value, cnty, *args):
        """Compute local features setbacks.

        This method will compute the setbacks using a county-specific
        regulations file that specifies either a static setback or a
        multiplier value that will be used along with the base setback
        distance to compute the setback.

        Parameters
        ----------
        regulation_value : float | int
            Setback distance in meters.
        cnty : geopandas.GeoDataFrame
            Regulations for a single county.
        features : geopandas.GeoDataFrame
            Features for the local county.
        feature_filter : callable
            A callable function that takes `features` and `cnty` as
            inputs and outputs a geopandas.GeoDataFrame with features
            clipped and/or localized to the input county.
        rasterizer : Rasterizer
            Instance of `Rasterizer` class used to rasterize the
            buffered county features.

        Returns
        -------
        setbacks : ndarray
            Raster array of setbacks
        """
        features, feature_filter, rasterizer = args
        logger.debug('- Computing setbacks for county FIPS {}'
                     .format(cnty.iloc[0]['FIPS']))
        log_mem(logger)
        features = feature_filter(features, cnty)
        negative_buffer = features.buffer(-1 * regulation_value)
        setbacks = features.buffer(0).difference(negative_buffer)
        return rasterizer.rasterize(list(setbacks))

    def compute_generic_exclusions(self, **__):
        """Compute generic setbacks.

        This method will compute the setbacks using a generic setback
        of `base_setback_dist * multiplier`.

        Returns
        -------
        setbacks : ndarray
            Raster array of setbacks
        """
        logger.info("Computing generic setbacks")
        if np.isclose(self._regulations.generic, 0):
            return self._rasterizer.rasterize(shapes=None)

        negative_buffer = self.features.buffer(-1 * self._regulations.generic)
        setbacks = self.features.buffer(0).difference(negative_buffer)
        return self._rasterizer.rasterize(list(setbacks))

    def _regulation_table_mask(self):
        """Return the regulation table mask for setback feature. """
        state = os.path.basename(self._features).split('.')[0]
        state = _get_state_name(state)
        states = self.regulations_table.State.apply(_get_state_name)
        states = states == state
        property_line = (self.regulations_table['Feature Type']
                         == 'property line')
        return states & property_line

    def parse_features(self):
        """Parse in parcel features.

        Warnings
        --------
        Use caution when calling this method, especially in multiple
        processes, as the returned feature files may be quite large.
        Reading 100 GB feature files in each of 36 sub-processes will
        quickly overwhelm your RAM.

        Returns
        -------
        `geopandas.GeoDataFrame`
            Geometries of features to setback from in exclusion
            coordinate system.
        """
        features = gpd.read_file(self._features)
        if features.crs is None:
            features = features.set_crs("EPSG:4326")
        return features.to_crs(crs=self._rasterizer.profile["crs"])


def _get_state_name(state):
    """Filter out non-alpha chars and casefold name"""
    return ''.join(filter(str.isalpha, state.casefold()))
