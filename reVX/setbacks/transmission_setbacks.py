# -*- coding: utf-8 -*-
"""
Compute setbacks exclusions
"""
import logging

from reVX.setbacks.base import AbstractBaseSetbacks, features_clipped_to_county

logger = logging.getLogger(__name__)


class TransmissionSetbacks(AbstractBaseSetbacks):
    """Transmission setbacks.

    These setbacks are computed against a single set of
    transmission features instead of against state level features.
    """

    @staticmethod
    def _feature_filter(features, cnty):
        """Filter the features given a county."""
        return features_clipped_to_county(features, cnty)

    def _regulation_table_mask(self):
        """Return the regulation table mask for setback feature. """
        return self.regulations_table['Feature Type'] == 'transmission'
