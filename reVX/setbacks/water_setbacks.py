# -*- coding: utf-8 -*-
"""
Compute setbacks exclusions
"""
from reVX.setbacks.base import BaseSetbacks, features_clipped_to_county


class WaterSetbacks(BaseSetbacks):
    """Water setbacks. """

    @staticmethod
    def _feature_filter(features, cnty):
        """Filter the features given a county."""
        return features_clipped_to_county(features, cnty)

    def _parse_regulations(self, regulations_fpath):
        """
        Parse water regulations, reduce table to just water features

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

        feats = regulations['Feature Type'].apply(str.strip).apply(str.lower)
        regulations = regulations.loc[feats == 'water']

        return regulations
