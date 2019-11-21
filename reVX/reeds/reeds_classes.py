# -*- coding: utf-8 -*-
"""
Bin ReEDS regions into 'resource' classes
"""
import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)


class ReEDSClasses:
    """
    Create ReEDS resource classes
    """
    def __init__(self, sc_table, class_bins, region_map='reeds_region'):
        """
        Parameters
        ----------
        sc_table : str | pandas.DataFrame
            Supply curve table, or path to file containing table
        class_bins : str | pandas.DataFrame
            Table of bins to use to create classes or path to file containing
            table of bins
        region_map : str | pandas.DataFrame | None
            Mapping of supply curve points to region to create classes for
        """
        self._sc_table = self._map_region(self._parse_table(sc_table),
                                          region_map)
        self._class_bins = self._parse_table(class_bins)

    @staticmethod
    def _parse_table(input_table):
        """
        Parse table from input argument

        Parameters
        ----------
        input_table : str | pandas.DataFrame
            Input table to parse

        Returns
        -------
        table : pandas.DataFrame
            Parsed table
        """
        table = input_table
        if isinstance(table, str):
            if table.endswith('.csv'):
                table = pd.read_csv(table)
            elif table.endwith('.json'):
                table = pd.read_json(table)
            else:
                msg = 'Cannot parse {}'.format(table)
                logger.error(msg)
                raise ValueError(msg)
        elif not isinstance(table, pd.DataFrame):
            msg = 'Cannot parse table from type {}'.format(type(table))
            logger.error(msg)
            raise ValueError(msg)

        return table

    @staticmethod
    def _map_region(sc_table, region_map):
        """
        Map regions to sc points and append to sc_table
        """
        if isinstance(region_map, str):
            if os.path.isfile(region_map):
                region_map = ReEDSClasses._parse_table(region_map)
            elif region_map in sc_table:
                region_map = sc_table[['sc_gid', region_map]].copy()
        elif not isinstance(region_map, pd.DataFrame):
            msg = ('Cannot parse region map from type {}'
                   .format(type(region_map)))
            logger.error(msg)
            raise ValueError(msg)
