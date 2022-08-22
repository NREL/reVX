# -*- coding: utf-8 -*-
"""
Compute setbacks exclusions
"""
import logging
import os
import geopandas as gpd

from rex.utilities import log_mem

from reVX.setbacks.base import BaseSetbacks


logger = logging.getLogger(__name__)


class ParcelSetbacks(BaseSetbacks):
    """
    Parcel setbacks - facilitates the use of negative buffers.
    This class uses duck typing to override `BaseSetbacks` behavior
    and should thus always be inherited alongside `BaseSetbacks`.
    """

    def _compute_generic_setbacks(self, features_fpath):
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
        logger.info("Computing generic setbacks")
        setback_features = self._parse_features(features_fpath)

        setbacks = [
            (geom, 1) for geom in setback_features.buffer(0).difference(
                setback_features.buffer(-1 * self._regulations.generic_setback)
            )
        ]

        return self._rasterizer.rasterize_setbacks(setbacks)

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

        setbacks = [
            (geom, 1) for geom in features.buffer(0).difference(
                features.buffer(-1 * setback)
            )
        ]

        return setbacks

    def _regulation_table_mask(self, features_fpath):
        """Return the regulation table mask for setback feature.

        Parameters
        ----------
        features_fpath : str
            Path to shape file with features to compute setbacks from.
            This file needs to have the state in the filename.
        """
        state = os.path.basename(features_fpath).split('.')[0]
        state = _get_state_name(state)
        states = self.regulations_table.State.apply(_get_state_name)
        states = states == state
        property_line = (self.regulations_table['Feature Type']
                         == 'property line')
        return states & property_line

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
        features = gpd.read_file(features_fpath)
        if features.crs is None:
            features = features.set_crs("EPSG:4326")
        return features.to_crs(crs=self.crs)


def _get_state_name(state):
    """Filter out non-alpha chars and casefold name"""
    return ''.join(filter(str.isalpha, state.casefold()))
