# -*- coding: utf-8 -*-
"""
Compute setbacks exclusions
"""
import logging
from warnings import warn

from reVX.setbacks.base import BaseSetbacks, features_clipped_to_county


logger = logging.getLogger(__name__)


class WaterSetbacks(BaseSetbacks):
    """Water setbacks. """

    @staticmethod
    def _feature_filter(features, cnty):
        """Filter the features given a county."""
        return features_clipped_to_county(features, cnty)

    def _pre_process_regulations(self, features_fpath):
        """
        Reduce regs to state corresponding to features_fpath if needed

        Parameters
        ----------
        features_fpath : str
            Path to shape file with features to compute setbacks from
        """
        mask = self.regulations_table['Feature Type'] == 'water'

        if not mask.any():
            msg = "Found no local regulations!"
            logger.warning(msg)
            warn(msg)

        self.regulations_table = (self.regulations_table.loc[mask]
                                  .reset_index(drop=True))
        super()._pre_process_regulations(features_fpath)
