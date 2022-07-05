# -*- coding: utf-8 -*-
"""
Compute setbacks exclusions
"""
import logging
import os
import geopandas as gpd

from rex.utilities import log_mem

from reVX.setbacks.base import BaseSetbacks
from reVX.setbacks.wind_setbacks import BaseWindSetbacks


logger = logging.getLogger(__name__)


# pylint: disable=no-member, too-few-public-methods
class _BaseParcelSetbacks:
    """
    Parcel setbacks - facilitates the use of negative buffers.
    This class uses duck typing to override `BaseSetbacks` behavior
    and should thus always be inherited alongside `BaseSetbacks`.
    """

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
        logger.info("Computing generic setbacks")
        setback_features = self._parse_features(features_fpath)

        setbacks = [
            (geom, 1) for geom in setback_features.buffer(0).difference(
                setback_features.buffer(-1 * self.generic_setback)
            )
        ]

        return self._rasterize_setbacks(setbacks)

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

    def _parse_regulations(self, regulations_fpath):
        """
        Parse parcel regulations, reduce table to just property lines

        Parameters
        ----------
        regulations_fpath : str
            Path to parcel regulations .csv file

        Returns
        -------
        regulations : pandas.DataFrame
            Parcel regulations table
        """
        regulations = super()._parse_regulations(regulations_fpath)

        mask = regulations['Feature Type'] == 'property line'
        regulations = regulations.loc[mask]

        return regulations

    def _check_regulations(self, features_fpath):
        """
        Reduce regs to state corresponding to features_fpath if needed.

        Parameters
        ----------
        features_fpath : str
            Path to shape file with features to compute setbacks from.
            This file needs to have the state in the filename.

        Returns
        -------
        regulations : geopandas.GeoDataFrame
            Parcel regulations
        """
        state = os.path.basename(features_fpath).split('.')[0]
        state = ''.join(filter(str.isalpha, state.lower()))

        regulation_states = self.regulations.State.apply(
            lambda s: ''.join(filter(str.isalpha, s.lower()))
        )

        mask = regulation_states == state
        regulations = self.regulations[mask].reset_index(drop=True)

        logger.debug(
            'Computing setbacks for parcel regulations in {} counties'
            .format(len(regulations))
        )

        return regulations

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



class SolarParcelSetbacks(_BaseParcelSetbacks, BaseSetbacks):
    """Solar Parcel Setbacks. """


class WindParcelSetbacks(_BaseParcelSetbacks, BaseWindSetbacks):
    """Wind Parcel Setbacks. """
