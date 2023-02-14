# -*- coding: utf-8 -*-
"""
Compute setbacks exclusions
"""
import logging
import os
import geopandas as gpd

from reVX.setbacks.base import AbstractBaseSetbacks


logger = logging.getLogger(__name__)


class ParcelSetbacks(AbstractBaseSetbacks):
    """Parcel setbacks - facilitates the use of negative buffers. """

    @staticmethod
    def _buffer(features, regulation_value):
        """Buffer parcels for county and return as list. """
        negative_buffer = features.buffer(-1 * regulation_value)
        return list(features.buffer(0).difference(negative_buffer))

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
