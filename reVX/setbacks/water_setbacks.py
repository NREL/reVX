# -*- coding: utf-8 -*-
"""
Compute setbacks exclusions
"""
from reVX.setbacks.base import BaseSetbacks


class WaterSetbacks(BaseSetbacks):
    """Water setbacks. """

    _FEATURE_FILE_EXTENSION = '.gpkg'

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

        mask = regulations['Feature Type'].apply(str.strip).lower() == 'water'
        regulations = regulations.loc[mask]

        return regulations
