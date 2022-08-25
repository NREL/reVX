# -*- coding: utf-8 -*-
"""
Compute setbacks exclusions
"""
import logging

from reVX.setbacks.base import AbstractBaseSetbacks, features_clipped_to_county


logger = logging.getLogger(__name__)


class WaterSetbacks(AbstractBaseSetbacks):
    """Water setbacks. """

    @staticmethod
    def _feature_filter(features, cnty):
        """Filter the features given a county."""
        return features_clipped_to_county(features, cnty)

    # pylint: disable=unused-argument
    def _regulation_table_mask(self, features_fpath):
        """Return the regulation table mask for setback feature.

        Parameters
        ----------
        features_fpath : str
            Path to shape file with features to compute setbacks from
        """
        return self.regulations_table['Feature Type'] == 'water'
