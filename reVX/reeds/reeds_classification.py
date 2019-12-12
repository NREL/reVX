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
    def __init__(self, rev_table, resource_classes, region_map='reeds_region',
                 sc_bins=5, cluster_kwargs={'cluster_on': 'trans_cap_cost',
                                            'method': 'kmeans'}):
        """
        Parameters
        ----------
        rev_table : str | pandas.DataFrame
            reV supply curve or aggregation table,
            or path to file containing table
        resource_classes : str | pandas.DataFrame | pandas.Series | dict
            Resource classes, either provided in a .csv, .json
            as a DataFrame or Series, or in a dictionary
        region_map : str | pandas.DataFrame
            Mapping of supply curve points to region to create classes for
         sc_bins : int
            Number of supply curve bins (clusters) to create for each
            region-class
        cluster_kwargs : dict
            kwargs for _cluster_classes
        """
        rev_table = self._parse_table(rev_table)
        rev_table = self._map_region(rev_table, region_map)
        rev_table = self._resource_classes(rev_table, resource_classes)
        self._rev_table = self._cluster_sc_bins(rev_table, sc_bins,
                                                **cluster_kwargs)
        self._groups = self._rev_table.groupby(['region', 'class', 'bin'])
        self._i = 0

    def __repr__(self):
        msg = ("{} contains {} region-class-class groups"
               .format(self.__class__.__name__, len(self)))
        return msg

    def __len__(self):
        return len(self._groups)

    def __getitem__(self, key):
        if key in self:
            group = self._groups.get_group(key)
        else:
            msg = "{} is an invalid group:\n{}".format(key, self.keys)
            logger.error(msg)
            raise ReedsKeyError(msg)

        return group

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self):
            self._i = 0
            raise StopIteration

        key = self.keys[self._i]
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
    def resource_classes(self):
        """
        Unique ReEDS resource classes

        Returns
        -------
        ndarray
        """
        return np.sort(self._rev_table['class'].unique())

    @property
    def sc_bins(self):
        """
        Unique ReEDS supply curve bins (clusters)

        Returns
        -------
        ndarray
        """
        return np.sort(self._rev_table['bin'].unique())

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
    def keys(self):
        """
        All unique group keys

        Returns
        -------
        list
        """
        return sorted(list(self._groups.groups.keys()))

    @property
    def region_class_bin_groups(self):
        """
        All unique (region, class, bin) groups

        Returns
        -------
        list
        """
        return self.keys

    @property
    def groups(self):
        """
        All unique group keys

        Returns
        -------
        list
        """
        return self.keys

    @property
    def aggregate_table(self):
        """
        Region, class, bin aggregate table

        Returns
        -------
        agg_table : pandas.DataFrame
        """
        cols = ['area_sq_km', 'capacity', 'trans_capacity']
        sum_table = self._groups[cols].sum()
        cols = ['latitude', 'longitude', 'mean_cf', 'mean_lcoe', 'mean_res',
                'trans_cap_cost', 'dist_mi', 'lcot', 'total_lcoe']
        mean_table = self._groups[cols].mean()
        agg_table = sum_table.join(mean_table)

        return agg_table.reset_index()

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
            for r, df in region_map.groupby(region_col):
                rev_table.loc[df['sc_gid'], 'region'] = r

            rev_table = rev_table.reset_index()

        return rev_table

    @staticmethod
    def _parse_class_bins(class_bins):
        """
        Parse resource class bins

        Parameters
        ----------
        class_bins : str | pandas.DataFrame | pandas.Series | dict
            Resource classes, either provided in a .csv, .json
            as a DataFrame or Series, or in a dictionary

        Returns
        -------
        attr : str
            reV table attribute (column) to bin
        class_bins : ndarray | list
            List / vector of bins to create classes from
        """
        if isinstance(class_bins, str):
            class_bins = ReedsClassifier._parse_table(class_bins)
        elif isinstance(class_bins, pd.Series):
            if not class_bins.name:
                msg = ('reV table attribute to bin not supplied as Series '
                       'name')
                logger.error(msg)
                raise ReedsValueError(msg)

            class_bins = class_bins.to_frame()
        elif isinstance(class_bins, dict):
            class_bins = pd.DataFrame(class_bins)
        elif not isinstance(class_bins, pd.DataFrame):
            msg = ('Cannot parse class bins from type {}'
                   .format(type(class_bins)))
            logger.error(msg)
            raise ReedsValueError(msg)

        attr = class_bins.columns
        if len(attr) > 1:
            msg = ('Can only bin classes on one attribute: {} were provided: '
                   '\n{}'.format(len(attr), attr))
            logger.error(msg)
            raise ReedsValueError(msg)

        attr = attr[0]
        class_bins = class_bins[attr].values

        return attr, class_bins

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
            trg_table['class'] = 1
            for _, df in trg_table.groupby('region'):
                df = df.sort_values('mean_lcoe')
                cum_sum = df['capacity'].cumsum()
                df.loc[:, 'class'] = pd.cut(x=cum_sum, bins=cap_breaks,
                                            labels=labels)
                classes.append(df)

            trg_table = pd.concat(classes)
        else:
            trg_table = trg_table.sort_values('mean_lcoe')
            cum_sum = trg_table['capacity'].cumsum()
            trg_table.loc[:, 'class'] = pd.cut(x=cum_sum, bins=cap_breaks,
                                               labels=labels)

        rev_table = rev_table.merge(trg_table[['sc_gid', 'class']],
                                    on='sc_gid', how='left')

        return rev_table

    @staticmethod
    def _resource_classes(rev_table, resource_classes):
        """
        Create resource classes

        Parameters
        ----------
        rev_table : pandas.DataFrame
            reV supply curve or aggregation table
        resource_classes : str | pandas.DataFrame | pandas.Series | dict
            Resource classes, either provided in a .csv, .json
            as a DataFrame or Series, or in a dictionary

        Returns
        -------
        rev_table : pandas.DataFrame
            Updated table with resource classes added
        """
        attr, class_bins = ReedsClassifier._parse_class_bins(resource_classes)

        if "TRG" in attr:
            rev_table = ReedsClassifier._TRG_bins(rev_table, class_bins)
        else:
            if attr not in rev_table:
                msg = ('{} is not a valid rev table attribute '
                       '(column header)'.format(attr))
                logger.error(msg)
                raise ReedsValueError(msg)

            labels = [i + 1 for i in range(len(class_bins) - 1)]
            rev_table['class'] = pd.cut(x=rev_table[attr],
                                        bins=class_bins, labels=labels)

        return rev_table

    @staticmethod
    def _cluster_sc_bins(rev_table, sc_bins, cluster_on='trans_cap_cost',
                         method='kmeans', **kwargs):
        """
        Create classes in each region-class group using given clustering method

        Parameters
        ----------
        rev_table : pandas.DataFrame
            reV supply curve or aggregation table
        sc_bins : int
            Number of supply curve bins (clusters) to create for each
            region-class
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
        labels = c_func(data, n_clusters=sc_bins,
                        **kwargs)
        if np.min(labels) == 0:
            labels = np.array(labels) + 1

        rev_table['bin'] = labels

        return rev_table

    @classmethod
    def create(cls, rev_table, resource_classes, region_map='reeds_region',
               sc_bins=5, cluster_kwargs={'cluster_on': 'trans_cap_cost',
                                          'method': 'kmeans'}):
        """
        Identify ReEDS regions and classes and dump and updated table

        Parameters
        ----------
        rev_table : str | pandas.DataFrame
            reV supply curve or aggregation table,
            or path to file containing table
        resource_classes : str | pandas.DataFrame | pandas.Series | dict
            Resource classes, either provided in a .csv, .json
            as a DataFrame or Series, or in a dictionary
        region_map : str | pandas.DataFrame
            Mapping of supply curve points to region to create classes for
        sc_bins : int
            Number of supply curve bins (clusters) to create for each
            region-class
        cluster_kwargs : dict
            kwargs for _cluster_classes

        Returns
        -------
        .table : pandas.DataFrame
            Updated table with region_id and class_bin columns
            added
        .aggregate_table : pandas.DataFrame
            Region, class, bin aggregate table
        """
        classes = cls(rev_table, resource_classes, region_map=region_map,
                      sc_bins=sc_bins, cluster_kwargs=cluster_kwargs)

        return classes.table, classes.aggregate_table
