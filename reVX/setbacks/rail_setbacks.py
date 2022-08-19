# -*- coding: utf-8 -*-
"""
Compute setbacks exclusions
"""
import logging

from reVX.setbacks.base import BaseSetbacks, features_clipped_to_county

logger = logging.getLogger(__name__)


class RailSetbacks(BaseSetbacks):
    """Rail setbacks.

    These setbacks are computed against a single set of railroad
    features instead of state level features.
    """

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
        mask = self.regulations_table['Feature Type'] == 'railroads'

        if not mask.any():
            msg = "Found no local regulations!"
            logger.error(msg)
            raise RuntimeError(msg)

        self.regulations_table = (self.regulations_table.loc[mask]
                                  .reset_index(drop=True))
        super()._pre_process_regulations(features_fpath)
