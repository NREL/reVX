# -*- coding: utf-8 -*-
"""
Compute setbacks exclusions
"""
import os
import re
import logging

from reVX.setbacks.base import BaseSetbacks

logger = logging.getLogger(__name__)


class StructureSetbacks(BaseSetbacks):
    """
    Structure setbacks
    """

    @staticmethod
    def _split_state_name(state_name):
        """
        Split state name at capitals to map .geojson files to
        regulations state names.

        Parameters
        ----------
        state_name : str
            State name from geojson files paths with out spaces

        Returns
        -------
        str
            State names with spaces added between Capitals (names) to
            match wind regulations state names.
        """
        state_name = ' '.join(a for a
                              in re.split(r'([A-Z][a-z]*)', state_name)
                              if a)

        return state_name

    @staticmethod
    def _get_feature_paths(features_fpath):
        """
        Find all structures .geojson files in structures dir

        Parameters
        ----------
        features_fpath : str
            Path to structures geojson for a single state, or directory
            containing geojsons for all states. Used to identify
            structures to build setbacks from. Files should be by state.

        Returns
        -------
        file_paths : list
            List of file paths to all structures .geojson files in
            structures_dir
        """
        is_file = (features_fpath.endswith('.geojson')
                   or features_fpath.endswith('.gpkg'))
        if is_file:
            file_paths = [features_fpath]
        else:
            file_paths = []
            for file in sorted(os.listdir(features_fpath)):
                is_file = file.endswith('.geojson') or file.endswith('.gpkg')
                if is_file:
                    file_paths.append(os.path.join(features_fpath, file))

        return file_paths

    def _check_regulations_table(self, features_fpath):
        """
        Reduce regs to state corresponding to features_fpath if needed

        Parameters
        ----------
        features_fpath : str
            Path to shape file with features to compute setbacks from
        """
        state_name = os.path.basename(features_fpath).split('.')[0]
        state = self._split_state_name(state_name)
        states = self.regulations_table["State"] == state
        structures = self.regulations_table['Feature Type'] == 'structures'
        not_ocb =  (self.regulations_table['Comment']
                    != 'Occupied Community Buildings')
        mask = (states & structures & not_ocb)

        if not mask.any():
            msg = ("There are no local regulations in {}!".format(state))
            logger.error(msg)
            raise RuntimeError(msg)

        self.regulations_table = (self.regulations_table.loc[mask]
                                  .reset_index(drop=True))
        super()._check_regulations_table(features_fpath)
