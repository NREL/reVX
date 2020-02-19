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

    TABLE_OUT_COLS = ('sc_gid', 'region', 'class', 'bin', 'capacity',
                      'mean_lcoe', 'trans_cap_cost', 'total_lcoe')

    AGG_TABLE_OUT_COLS = ('region', 'class', 'bin', 'capacity',
                          'trans_cap_cost', 'dist_mi')

    def __init__(self, rev_table, resource_classes, region_map='reeds_region',
                 sc_bins=5, cluster_kwargs={'cluster_on': 'trans_cap_cost',
                                            'method': 'kmeans', 'norm': None},
                 filter=None, trg_by_region=False):
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
            kwargs for _cluster_sc_bins and underlying clustering method
        filter : dict | NoneType
            Column value pair(s) to filter on. If None don't filter
        trg_by_region : bool
            Groupby on region when computing TRGs
        """
        rev_table = self._parse_table(rev_table)
        if filter is not None:
            for col, v in filter.items():
                mask = rev_table[col] == v
                rev_table = rev_table.loc[mask]

        rev_table = self._map_region(rev_table, region_map)
        rev_table = self._resource_classes(rev_table, resource_classes,
                                           trg_by_region=trg_by_region)
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
    def table_slim(self):
        """
        Supply curve or aggregation table with only columns in TABLE_OUT_COLS

        Returns
        -------
        pandas.DataFrame
        """
        cols = [c for c in self.TABLE_OUT_COLS if c in self.table]
        return self.table[cols]

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

    @property
    def aggregate_table_slim(self):
        """
        Aggregate table with only columns in AGG_TABLE_OUT_COLS

        Returns
        -------
        agg_table : pandas.DataFrame
        """
        agg_table = self.aggregate_table
        cols = [c for c in agg_table if c in self.AGG_TABLE_OUT_COLS]
        return agg_table[cols]

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
            rev_table['region'] = 1
        else:
            region_map = ReedsClassifier._parse_region_map(region_map,
                                                           rev_table)

            if 'sc_gid' not in region_map:
                merge_cols = [c for c in region_map.columns if c in rev_table]
                if not merge_cols:
                    msg = ('region map must contain a "sc_gid" column or a '
                           'column in common with the Supply Curve table.')
                    logger.error(msg)
                    raise ReedsValueError(msg)

                region_map = pd.merge(rev_table[['sc_gid', ] + merge_cols],
                                      region_map, on=merge_cols)

            region_col = [c for c in region_map.columns if c != 'sc_gid']
            rev_table['region'] = None
            rev_table = rev_table.set_index('sc_gid')
            for r, df in region_map.groupby(region_col):
                rev_table.loc[df['sc_gid'], 'region'] = r

            mask = ~rev_table['region'].isnull()
            rev_table = rev_table.loc[mask].reset_index()

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
        class_bins : pandas.DataFrame
            DataFrame of Resource classifiers:
            - range bins
            - TRG capacity bins
            - numerical bins
            - catagorical bins
        """
        class_bins = ReedsClassifier._parse_table(class_bins)

        if 'class' in class_bins:
            class_bins = class_bins.set_index('class')

        return class_bins

    # flake8: noqa
    # noqa: C901
    @staticmethod
    def _parse_classifiers(class_bins):
        """
        [summary]

        Parameters
        ----------
        class_bins : pandas.DataFrame
            DataFrame of Resource classifiers:
            - range bins
            - TRG capacity bins
            - numerical bins
            - catagorical bins

        Returns
        -------
        classifiers : list
            List of classifiers to run. Each classifier is a dict:
            {'group': group, 'bins': {method: input data}}
        groupby : NoneType | list
            Columns to run groupby on

        Raises
        ------
        RuntimeError
            Runtime error in input columns are not proper for binning
            classes
        """
        cat_cols = [c for c, dtype in class_bins.dtypes.iteritems()
                    if not np.issubdtype(dtype, np.number)]

        bin_cols = [c for c in class_bins if c not in cat_cols]
        range_cols = None
        trg_col = None
        bin_col = None
        if cat_cols:
            groupby = cat_cols
        else:
            groupby = None

        n_bins = len(bin_cols)
        if not n_bins and cat_cols:
            if not len(np.unique(class_bins[cat_cols])) == len(class_bins):
                groupby = None
            else:
                msg = ("Catagorical bins must have the same number of bins "
                       "as classes: {}".format(class_bins[cat_cols]))
                logger.error(msg)
                raise RuntimeError(msg)

        elif n_bins > 2:
            msg = ("To many class bins have been provided: {}. "
                   "Along with catagorical bins, only provide:"
                   "\n a set of range bin (*_min, *_max), TRG_cap, "
                   "or numerical bins".format(bin_cols))
            logger.error(msg)
            raise RuntimeError(msg)

        elif n_bins == 2:
            if "TRG_cap" in bin_cols and len(bin_cols) > 1:
                msg = ("TRG bins cannot be paired with other types of class "
                       "bins: {}".format(bin_cols))
                logger.error(msg)
                raise RuntimeError(msg)

            range_cols = ['_'.join(c.split('_')[:-1])
                          for c in bin_cols if c.endswith(('min', 'max'))]
            if len(range_cols) != 2:
                msg = ("Minimun ({}_min) and maximum ({}_max) values must be "
                       "provided for range bins".format(range_cols[0]))
                logger.error(msg)
                raise RuntimeError(msg)

        else:
            if 'TRG_cap' in bin_cols:
                trg_col = ['TRG_cap']
            else:
                bin_col = bin_cols

        if groupby is not None:
            classifiers = []
            for cat, df in class_bins.groupby(groupby):
                if range_cols is not None:
                    method = (ReedsClassifier._range_classes, df[range_cols])
                elif trg_col is not None:
                    method = (ReedsClassifier._TRG_classes, df[trg_col])
                elif bin_col is not None:
                    method = (ReedsClassifier._bin_classes, df[bin_col])

                classifiers.append({'group': cat, 'bins': method})
        else:
            if range_cols is not None:
                method = (ReedsClassifier._range_classes, class_bins)
            elif trg_col is not None:
                method = (ReedsClassifier._TRG_classes, class_bins)
            elif bin_col is not None:
                method = (ReedsClassifier._bin_classes, class_bins)
            else:
                method = (ReedsClassifier._catagorical_classes, class_bins)

            classifiers = method

        return classifiers, groupby

    @staticmethod
    def _TRG_classes(rev_table, trg_bins, by_region=False):
        """
        Create TRG (technical resource groups) using given cummulative
        capacity bin widths

        Parameters
        ----------
        rev_table : pandas.DataFrame
            reV supply curve or aggregation table
        trg_bins : pandas.Series
            Cummulative capacity bin widths to create TRGs from
            (in GW)
        by_region : bool
            Groupby on region

        Returns
        -------
        rev_table : pandas.DataFrame
            Updated table with TRG classes added
        """
        cap_breaks = np.cumsum(trg_bins['TRG_cap'].values) * 1000  # GW to MW
        cap_breaks = np.concatenate(([0., ], cap_breaks, [float('inf')]),
                                    axis=0)
        labels = trg_bins.index.values

        cols = ['sc_gid', 'capacity', 'mean_lcoe', 'region']
        trg_classes = rev_table[cols].copy()
        if by_region:
            classes = []
            trg_classes['class'] = 1
            for _, df in trg_classes.groupby('region'):
                df = df.sort_values('mean_lcoe')
                cum_sum = df['capacity'].cumsum()
                df.loc[:, 'class'] = pd.cut(x=cum_sum, bins=cap_breaks,
                                            labels=labels)
                classes.append(df)

            trg_classes = pd.concat(classes)
        else:
            trg_classes = trg_classes.sort_values('mean_lcoe')
            cum_sum = trg_classes['capacity'].cumsum()
            trg_classes.loc[:, 'class'] = pd.cut(x=cum_sum, bins=cap_breaks,
                                                 labels=labels)

        rev_table = rev_table.merge(trg_classes[['sc_gid', 'class']],
                                    on='sc_gid', how='left')

        return rev_table

    @staticmethod
    def _range_classes(rev_table, range_bins):
        """
        [summary]

        Parameters
        ----------
        rev_table : [type]
            [description]
        range_bins : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        sc_col = '_'.join(range_bins.columns[0].split('_')[:-1])
        cols = ['{}_min'.format(sc_col), '{}_max'.format(sc_col)]
        bins = range_bins[cols].values
        bins = pd.IntervalIndex.from_arrays(bins[:, 0], bins[:, 1])
        labels = range_bins.index

        rev_table['class'] = pd.cut(x=rev_table[sc_col], bins=bins,
                                    labels=labels)

        return rev_table

    @staticmethod
    def _bin_classes(rev_table, class_bins):
        """
        [summary]

        Parameters
        ----------
        rev_table : [type]
            [description]
        class_bins : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        sc_col = class_bins.columns[0]
        bins = class_bins.values
        idx = np.argsort(bins)
        bins = bins[idx]
        labels = class_bins.index.values[idx]

        rev_table['class'] = pd.cut(x=rev_table[sc_col],
                                    bins=bins, labels=labels)

        return rev_table

    @staticmethod
    def _catagorical_classes(rev_table, cat_bins):
        """
        [summary]

        Parameters
        ----------
        rev_table : [type]
            [description]
        cat_bins : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        sc_col = cat_bins.columns[0]
        cat_bins = cat_bins.reset_index()
        rev_table = rev_table.merge(cat_bins, on=sc_col, how='left')

        return rev_table

    @staticmethod
    def _resource_classes(rev_table, resource_classes):
        """
        Create resource classes

        Parameters
        ----------
        rev_table : pandas.DataFrame
            reV supply curve or aggregation table
        resource_classes : str | pandas.DataFrame
            Resource classes, either provided in a .csv, .json
            as a DataFrame or Series, or in a dictionary
        trg_by_region : bool
            Groupby on region for TRGs

        Returns
        -------
        rev_table : pandas.DataFrame
            Updated table with resource classes added
        """
        classifiers = ReedsClassifier._parse_class_bins(resource_classes)
        classifiers, groupby = ReedsClassifier._parse_classifiers(classifiers)

        if groupby is not None:
            class_tables = []
            groups = rev_table.groupby(groupby)
            for classifier in classifiers:
                group_table = groups.get_group(classifier['group'])
                func, args = classifier['bins']
                class_tables.append(func(group_table, args))

            labels = np.array([i + 1 for i in range(len(class_bins) - 1)])
            idx = np.argsort(class_bins)
            class_bins = class_bins[idx]
            idx = [i for i in idx if i < len(labels)]
            labels = labels[idx]

            rev_table['class'] = pd.cut(x=rev_table[attr],
                                        bins=class_bins, labels=labels)

        return rev_table

    @staticmethod
    def _cluster_sc_bins(rev_table, sc_bins, cluster_on='trans_cap_cost',
                         method='kmeans', norm=None, **kwargs):
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
        norm : str
            Normalization method to use (see sklearn.preprocessing.normalize)
            if None range normalize
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
        data = func(rev_table[cluster_on].values, norm=norm)
        labels = c_func(data, n_clusters=sc_bins,
                        **kwargs)
        if np.min(labels) == 0:
            labels = np.array(labels) + 1

        rev_table['bin'] = labels

        return rev_table

    @classmethod
    def create(cls, rev_table, resource_classes, region_map='reeds_region',
               sc_bins=5, cluster_kwargs={'cluster_on': 'trans_cap_cost',
                                          'method': 'kmeans', 'norm': None},
               filter=None, trg_by_region=False):
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
        filter : dict | NoneType
            Column value pair(s) to filter on. If None don't filter
        trg_by_region : bool
            Groupby on region when computing TRGs

        Returns
        -------
        .table : pandas.DataFrame
            Updated table with region_id and class_bin columns
            added. Includes all columns.
        .table_slim : pandas.DataFrame
            Updated table with region_id and class_bin columns
            added. Only includes columns in TABLE_OUT_COLS.
        .aggregate_table : pandas.DataFrame
            Region, class, bin aggregate table. Includes all columns.
        .aggregate_table_slim : pandas.DataFrame
            Region, class, bin aggregate table. Only inlcudes columns in
            AGG_TABLE_OUT_COLS.
        """
        classes = cls(rev_table, resource_classes, region_map=region_map,
                      sc_bins=sc_bins, cluster_kwargs=cluster_kwargs,
                      filter=filter, trg_by_region=trg_by_region)
        out = (classes.table, classes.table_slim, classes.aggregate_table,
               classes.aggregate_table_slim)

        return out
