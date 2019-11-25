# -*- coding: utf-8 -*-
"""
Map ReEDS geographic regions and classes to Supply Curve points
"""
import logging
import numpy as np
import os
import pandas as pd

from reVX.utilities.exceptions import ReEDSValueError, ReEDSKeyError

logger = logging.getLogger(__name__)


class ReEDSClasses:
    """
    Create ReEDS resource classes
    """
    def __init__(self, rev_table, class_bins, region_map='reeds_region'):
        """
        Parameters
        ----------
        rev_table : str | pandas.DataFrame
            reV supply curve or aggregation table,
            or path to file containing table
        class_bins : str | pandas.DataFrame | pandas.Series | dict
            Bins to use for creating classes, either provided in a .csv, .json
            as a DataFrame or Series, or in a dictionary
        region_map : str | pandas.DataFrame
            Mapping of supply curve points to region to create classes for
        """
        rev_table = self._parse_table(rev_table)
        rev_table = self._map_region(rev_table, region_map)
        self._rev_table = self._bin_classes(rev_table, class_bins)
        self._groups = self._rev_table.groupby(['region_id', 'class_bin'])
        self._i = 0

    def __repr__(self):
        msg = ("{} contains {} region-class groups"
               .format(self.__class__.__name__, len(self)))
        return msg

    def __len__(self):
        return len(self._groups)

    def __getitem__(self, key):
        if key in self:
            group = self._groups.get_group(key)

        return group

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self):
            self._i = 0
            raise StopIteration

        key = self.region_class_pairs[self._i]
        group = self[key]
        self._i += 1

        return group

    def __contains__(self, key):
        test = key in self._groups.groups
        if not test:
            msg = "{} does not exist in {}".format(key, self)
            raise ReEDSKeyError(msg)

        return test

    @property
    def regions(self):
        """
        Unique ReEDS geographic regions

        Returns
        -------
        ndarray
        """
        return np.sort(self._rev_table['region_id'].unique())

    @property
    def classes(self):
        """
        Unique ReEDS class bins

        Returns
        -------
        ndarray
        """
        return np.sort(self._rev_table['class_bin'].unique())

    @property
    def table(self):
        """
        Supply curve or aggregation table

        Returns
        -------
        pandas.DataFrame
        """
        return self._rev_table

    @property
    def region_class_pairs(self):
        """
        All unique (region, class) pairs

        Returns
        -------
        list
        """
        return sorted(list(self._groups.groups.keys()))

    @property
    def region_class_groups(self):
        """
        Region class groupby object

        Returns
        -------
        pandas.groupby
        """
        return self._groups

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
    def _parse_region_map(region_map, rev_table):
        """
        Parse region map from input arg

        Parameters
        ----------
        region_map : str | pandas.DataFrame | None
            Mapping of supply curve points to region to create classes for
        rev_table : pandas.DataFrame
            reV supply curve or aggregation table

        Returns
        -------
        region_map : pandas.DataFrame
            Mapping of region to sc_gid
        """
        if isinstance(region_map, str):
            if os.path.isfile(region_map):
                region_map = ReEDSClasses._parse_table(region_map)
            elif region_map in rev_table:
                region_map = rev_table[['sc_gid', region_map]].copy()
            else:
                msg = ('{} is not a valid file path or reV table '
                       'column label'.format(type(region_map)))
                logger.error(msg)
                raise ReEDSValueError(msg)
        elif not isinstance(region_map, pd.DataFrame):
            msg = ('Cannot parse region map from type {}'
                   .format(type(region_map)))
            logger.error(msg)
            raise ReEDSValueError(msg)

        return region_map

    @staticmethod
    def _map_region(rev_table, region_map=None):
        """
        Map regions to sc points and append to rev_table

        Parameters
        ----------
        rev_table : pandas.DataFrame
            reV supply curve or aggregation table
        region_map : str | pandas.DataFrame | None
            Mapping of supply curve points to region to create classes for


        Returns
        -------
        rev_table : pandas.DataFrame
            Updated table with region_id added
        """
        if region_map is None:
            rev_table['region_id'] = 0
        else:
            region_map = ReEDSClasses._parse_region_map(region_map, rev_table)

            if 'sc_gid' not in region_map:
                msg = ('region map must contain a "sc_gid" column to allow '
                       'mapping to Supply Curve table')
                logger.error(msg)
                raise ReEDSValueError(msg)

            region_col = [c for c in region_map.columns if c != 'sc_gid']
            rev_table['region_id'] = 0
            rev_table = rev_table.set_index('sc_gid')
            for i, (_, df) in enumerate(region_map.groupby(region_col)):
                rev_table.loc[df['sc_gid'], 'region_id'] = i

            rev_table = rev_table.reset_index()

        return rev_table

    @staticmethod
    def _parse_class_bins(class_bins):
        """
        Parse bins needed to create classes

        Parameters
        ----------
        class_bins : str | pandas.DataFrame | pandas.Series | dict
            Bins to use for creating classes, either provided in a .csv, .json
            as a DataFrame or Series, or in a dictionary

        Returns
        -------
        attr : str
            reV table attribute (column) to bin
        bins : ndarray | list
            List / vector of bins to create classes from
        """
        if isinstance(class_bins, str):
            class_bins = ReEDSClasses._parse_table(class_bins)
        elif isinstance(class_bins, pd.Series):
            if not class_bins.name:
                msg = ('reV table attribute to bin not supplied as Series '
                       'name')
                logger.error(msg)
                raise ReEDSValueError(msg)

            class_bins = class_bins.to_frame()
        elif isinstance(class_bins, dict):
            class_bins = pd.DataFrame(class_bins)
        elif not isinstance(class_bins, pd.DataFrame):
            msg = ('Cannot parse class bins from type {}'
                   .format(type(class_bins)))
            logger.error(msg)
            raise ReEDSValueError(msg)

        attr = class_bins.columns
        if len(attr) > 1:
            msg = ('Can only bin classes on one attribute: {} were provided: '
                   '\n{}'.format(len(attr), attr))
            logger.error(msg)
            raise ReEDSValueError(msg)

        attr = attr[0]
        bins = class_bins[attr].values

        return attr, bins

    @staticmethod
    def _TRG_classes(rev_table, trg_bins, by_region=True):
        """
        Create TRG (technical resource groups) using given cummulative
        capacity bin widths

        Parameters
        ----------
        rev_table : pandas.DataFrame
            reV supply curve or aggregation table
        trg_bins : list | ndarray
            Cummulative capacity bin widths to create TRGs from
            (in GW)

        Returns
        -------
        rev_table : pandas.DataFrame
            Updated table with TRG classes added
        """
        cap_breaks = np.cumsum(trg_bins) * 1000  # convert to MW
        cap_breaks = np.concatenate(([0., ], cap_breaks, [float('inf')]),
                                    axis=0)
        labels = [i + 1 for i in range(len(cap_breaks) - 1)]

        cols = ['sc_gid', 'capacity', 'mean_lcoe', 'region_id']
        trg_table = rev_table[cols].copy()
        if by_region:
            classes = []
            trg_table['class_bin'] = 0
            for _, df in trg_table.groupby('region_id'):
                df = df.sort_values('mean_lcoe')
                cum_sum = df['capacity'].cumsum()
                df.loc[:, 'class_bin'] = pd.cut(x=cum_sum, bins=cap_breaks,
                                                labels=labels)
                classes.append(df)

            trg_table = pd.concat(classes)
        else:
            trg_table = trg_table.sort_values('mean_lcoe')
            cum_sum = trg_table['capacity'].cumsum()
            trg_table.loc[:, 'class_bin'] = pd.cut(x=cum_sum, bins=cap_breaks,
                                                   labels=labels)

        rev_table = rev_table.merge(trg_table[['sc_gid', 'class_bin']],
                                    on='sc_gid', how='left')

        return rev_table

    @staticmethod
    def _bin_classes(rev_table, class_bins):
        """
        Bin sites in to classes

        Parameters
        ----------
        rev_table : pandas.DataFrame
            reV supply curve or aggregation table
        class_bins : str | pandas.DataFrame | pandas.Series | dict
            Bins to use for creating classes, either provided in a .csv, .json
            as a DataFrame or Series, or in a dictionary

        Returns
        -------
        rev_table : pandas.DataFrame
            Updated table with class bins added
        """
        attr, bins = ReEDSClasses._parse_class_bins(class_bins)

        if "TRG" in attr:
            rev_table = ReEDSClasses._TRG_classes(rev_table, bins)
        else:
            if attr not in rev_table:
                msg = ('{} is not a valid rev table attribute '
                       '(column header)'.format(attr))
                logger.error(msg)
                raise ReEDSValueError(msg)

            labels = [i + 1 for i in range(len(bins) - 1)]
            rev_table['class_bin'] = pd.cut(x=rev_table[attr],
                                            bins=bins, labels=labels)

        return rev_table

    @classmethod
    def create(cls, rev_table, class_bins, region_map='reeds_region'):
        """
        Identify ReEDS regions and classes and dump and updated table

        Parameters
        ----------
        rev_table : str | pandas.DataFrame
            reV supply curve or aggregation table,
            or path to file containing table
        class_bins : str | pandas.DataFrame | pandas.Series | dict
            Bins to use for creating classes, either provided in a .csv, .json
            as a DataFrame or Series, or in a dictionary
        region_map : str | pandas.DataFrame
            Mapping of supply curve points to region to create classes for

        Returns
        -------
        rev_table : pandas.DataFrame
            Updated table with region_id and class_bin columns
            added
        """
        classes = cls(rev_table, class_bins, region_map=region_map)
        return classes.table
