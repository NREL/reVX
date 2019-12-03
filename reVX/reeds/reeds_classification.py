# -*- coding: utf-8 -*-
"""
Map ReEDS geographic regions and classes to Supply Curve points
"""
import logging
import numpy as np
import os
import pandas as pd

from reVX.utilities.cluster_methods import ClusteringMethods
from reVX.utilities.exceptions import ReedsValueError, ReedsKeyError

logger = logging.getLogger(__name__)


class ReedsClassifier:
    """
    Create ReEDS resource classes
    """
    def __init__(self, rev_table, bins, region_map='reeds_region', classes=3,
                 cluster_kwargs={'cluster_on': 'trans_cap_cost',
                                 'method': 'kmeans'}):
        """
        Parameters
        ----------
        rev_table : str | pandas.DataFrame
            reV supply curve or aggregation table,
            or path to file containing table
        bins : str | pandas.DataFrame | pandas.Series | dict
            Resource bins, either provided in a .csv, .json
            as a DataFrame or Series, or in a dictionary
        region_map : str | pandas.DataFrame
            Mapping of supply curve points to region to create classes for
        classes : int
            Number of classes (clusters) to create for each region-bin
        cluster_kwargs : dict
            kwargs for _cluster_classes
        """
        rev_table = self._parse_table(rev_table)
        rev_table = self._map_region(rev_table, region_map)
        rev_table = self._resource_bins(rev_table, bins)
        self._rev_table = self._cluster_classes(rev_table, classes,
                                                **cluster_kwargs)
        self._groups = self._rev_table.groupby(['region', 'bin', 'class'])
        self._i = 0

    def __repr__(self):
        msg = ("{} contains {} region-bin-class groups"
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

        key = self.region_bin_class_groups[self._i]
        group = self[key]
        self._i += 1

        return group

    def __contains__(self, key):
        test = key in self._groups.groups
        if not test:
            msg = "{} does not exist in {}".format(key, self)
            raise ReedsKeyError(msg)

        return test

    @property
    def regions(self):
        """
        Unique ReEDS geographic regions

        Returns
        -------
        ndarray
        """
        return np.sort(self._rev_table['region'].unique())

    @property
    def bins(self):
        """
        Unique ReEDS resource bins

        Returns
        -------
        ndarray
        """
        return np.sort(self._rev_table['bin'].unique())

    @property
    def classes(self):
        """
        Unique ReEDS classes (clusters)

        Returns
        -------
        ndarray
        """
        return np.sort(self._rev_table['class'].unique())

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
    def region_bin_class_groups(self):
        """
        All unique (region, bin, class) groups

        Returns
        -------
        list
        """
        return sorted(list(self._groups.groups.keys()))

    @property
    def aggregate_table(self):
        """
        Region, bin, class aggregate table

        Returns
        -------
        agg_table : pandas.DataFrame
        """
        cols = ['area_sq_km', 'capacity', 'latitude', 'longitude', 'mean_cf',
                'mean_lcoe', 'mean_res', 'pct_slope', 'trans_capacity',
                'trans_cap_cost', 'dist_mi', 'lcot', 'total_lcoe']
        agg_table = self._groups.sum()
        return agg_table[cols].reset_index()

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
                region_map = ReedsClassifier._parse_table(region_map)
            elif region_map in rev_table:
                region_map = rev_table[['sc_gid', region_map]].copy()
            else:
                msg = ('{} is not a valid file path or reV table '
                       'column label'.format(type(region_map)))
                logger.error(msg)
                raise ReedsValueError(msg)
        elif not isinstance(region_map, pd.DataFrame):
            msg = ('Cannot parse region map from type {}'
                   .format(type(region_map)))
            logger.error(msg)
            raise ReedsValueError(msg)

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
            rev_table['region'] = 0
        else:
            region_map = ReedsClassifier._parse_region_map(region_map,
                                                           rev_table)

            if 'sc_gid' not in region_map:
                msg = ('region map must contain a "sc_gid" column to allow '
                       'mapping to Supply Curve table')
                logger.error(msg)
                raise ReedsValueError(msg)

            region_col = [c for c in region_map.columns if c != 'sc_gid']
            rev_table['region'] = 0
            rev_table = rev_table.set_index('sc_gid')
            for i, (_, df) in enumerate(region_map.groupby(region_col)):
                rev_table.loc[df['sc_gid'], 'region'] = i

            rev_table = rev_table.reset_index()

        return rev_table

    @staticmethod
    def _parse_bins(bins):
        """
        Parse bins needed to create classes

        Parameters
        ----------
        bins : str | pandas.DataFrame | pandas.Series | dict
            Resource bins, either provided in a .csv, .json
            as a DataFrame or Series, or in a dictionary

        Returns
        -------
        attr : str
            reV table attribute (column) to bin
        bins : ndarray | list
            List / vector of bins to create classes from
        """
        if isinstance(bins, str):
            bins = ReedsClassifier._parse_table(bins)
        elif isinstance(bins, pd.Series):
            if not bins.name:
                msg = ('reV table attribute to bin not supplied as Series '
                       'name')
                logger.error(msg)
                raise ReedsValueError(msg)

            bins = bins.to_frame()
        elif isinstance(bins, dict):
            bins = pd.DataFrame(bins)
        elif not isinstance(bins, pd.DataFrame):
            msg = ('Cannot parse class bins from type {}'
                   .format(type(bins)))
            logger.error(msg)
            raise ReedsValueError(msg)

        attr = bins.columns
        if len(attr) > 1:
            msg = ('Can only bin classes on one attribute: {} were provided: '
                   '\n{}'.format(len(attr), attr))
            logger.error(msg)
            raise ReedsValueError(msg)

        attr = attr[0]
        bins = bins[attr].values

        return attr, bins

    @staticmethod
    def _TRG_bins(rev_table, trg_bins, by_region=True):
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

        cols = ['sc_gid', 'capacity', 'mean_lcoe', 'region']
        trg_table = rev_table[cols].copy()
        if by_region:
            classes = []
            trg_table['bin'] = 0
            for _, df in trg_table.groupby('region'):
                df = df.sort_values('mean_lcoe')
                cum_sum = df['capacity'].cumsum()
                df.loc[:, 'bin'] = pd.cut(x=cum_sum, bins=cap_breaks,
                                          labels=labels)
                classes.append(df)

            trg_table = pd.concat(classes)
        else:
            trg_table = trg_table.sort_values('mean_lcoe')
            cum_sum = trg_table['capacity'].cumsum()
            trg_table.loc[:, 'bin'] = pd.cut(x=cum_sum, bins=cap_breaks,
                                             labels=labels)

        rev_table = rev_table.merge(trg_table[['sc_gid', 'bin']],
                                    on='sc_gid', how='left')

        return rev_table

    @staticmethod
    def _resource_bins(rev_table, bins):
        """
        Create resource bins

        Parameters
        ----------
        rev_table : pandas.DataFrame
            reV supply curve or aggregation table
        bins : str | pandas.DataFrame | pandas.Series | dict
            Resource bins, either provided in a .csv, .json
            as a DataFrame or Series, or in a dictionary

        Returns
        -------
        rev_table : pandas.DataFrame
            Updated table with resource bins added
        """
        attr, bins = ReedsClassifier._parse_bins(bins)

        if "TRG" in attr:
            rev_table = ReedsClassifier._TRG_bins(rev_table, bins)
        else:
            if attr not in rev_table:
                msg = ('{} is not a valid rev table attribute '
                       '(column header)'.format(attr))
                logger.error(msg)
                raise ReedsValueError(msg)

            labels = [i + 1 for i in range(len(bins) - 1)]
            rev_table['bin'] = pd.cut(x=rev_table[attr],
                                      bins=bins, labels=labels)

        return rev_table

    @staticmethod
    def _cluster_classes(rev_table, classes, cluster_on='trans_cap_cost',
                         method='kmeans', **kwargs):
        """
        Create classes in each region-bin group using given clustering method

        Parameters
        ----------
        rev_table : pandas.DataFrame
            reV supply curve or aggregation table
        classes : int
            Number of classes (clusters) to create for each region-bin
        cluster_on : str | list
            Columns in rev_table to cluster on
        method : str
            Clustering method to use for creating classes
        kwargs : dict
            kwargs for clustering method

        Returns
        -------
        rev_table : pandas.DataFrame
            Updated table with classes
        """
        c_func = getattr(ClusteringMethods, method)

        if isinstance(cluster_on, str):
            cluster_on = [cluster_on, ]

        func = ClusteringMethods._normalize_values
        data = func(rev_table[cluster_on].values, **kwargs)
        labels = c_func(data, n_clusters=classes,
                        **kwargs)
        rev_table['class'] = labels

        return rev_table

    @classmethod
    def create(cls, rev_table, bins, region_map='reeds_region', classes=3,
               cluster_kwargs={'cluster_on': 'trans_cap_cost',
                               'method': 'kmeans'}):
        """
        Identify ReEDS regions and classes and dump and updated table

        Parameters
        ----------
        rev_table : str | pandas.DataFrame
            reV supply curve or aggregation table,
            or path to file containing table
        bins : str | pandas.DataFrame | pandas.Series | dict
            Resource bins, either provided in a .csv, .json
            as a DataFrame or Series, or in a dictionary
        region_map : str | pandas.DataFrame
            Mapping of supply curve points to region to create classes for
        classes : int
            Number of classes (clusters) to create for each region-bin
        cluster_kwargs : dict
            kwargs for _cluster_classes

        Returns
        -------
        rev_table : pandas.DataFrame
            Updated table with region_id and class_bin columns
            added
        """
        classes = cls(rev_table, bins, region_map=region_map, classes=classes,
                      cluster_kwargs=cluster_kwargs)
        return classes.table
