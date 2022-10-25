# -*- coding: utf-8 -*-
"""
Compute setbacks exclusions
"""
import os
import logging
import fiona
import geopandas as gpd

from reVX.setbacks.base import AbstractBaseSetbacks
from reVX.utilities.utilities import STATES_ABBR_MAP

logger = logging.getLogger(__name__)


class RoadSetbacks(AbstractBaseSetbacks):
    """
    Road setbacks
    """

    def parse_features(self):
        """
        Load roads from gdb file, convert to exclusions coordinate
        system.

        Warnings
        --------
        Use caution when calling this method, especially in multiple
        processes, as the returned feature files may be quite large.
        Reading 100 GB feature files in each of 36 sub-processes will
        quickly overwhelm your RAM.

        Returns
        -------
        roads : `geopandas.GeoDataFrame.sindex`
            Geometries for roads in gdb file, in exclusion coordinate
            system
        """
        lyr = fiona.listlayers(self._features)[0]
        roads = gpd.read_file(self._features, driver='FileGDB', layer=lyr)
        return roads.to_crs(crs=self._rasterizer.profile["crs"])

    @staticmethod
    def get_feature_paths(features_fpath):
        """
        Find all roads gdb files in roads_dir

        Parameters
        ----------
        features_fpath : str
            Path to state here streets gdb file or directory containing
            states gdb files. Used to identify roads to build setbacks
            from. Files should be by state.

        Returns
        -------
        file_paths : list
            List of file paths to all roads .gdp files in roads_dir
        """
        is_file = (features_fpath.endswith('.gdb')
                   or features_fpath.endswith('.gpkg'))
        if is_file:
            file_paths = [features_fpath]
        else:
            file_paths = []
            for file in sorted(os.listdir(features_fpath)):
                is_file = file.endswith('.gdb') or file.endswith('.gpkg')
                if is_file and file.startswith('Streets_USA'):
                    file_paths.append(os.path.join(features_fpath, file))

        return file_paths

    def _regulation_table_mask(self):
        """Return the regulation table mask for setback feature. """
        state = self._features.split('.')[0].split('_')[-1]
        if 'Abbr' not in self.regulations_table:
            states = self.regulations_table['State'].str.title()
            self.regulations_table['Abbr'] = states.map(STATES_ABBR_MAP)
        states = self.regulations_table['Abbr'] == state

        feature_types = {'roads', 'highways', 'highways 111'}
        features = self.regulations_table['Feature Type'].isin(feature_types)
        return states & features
