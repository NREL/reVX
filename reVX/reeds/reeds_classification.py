# -*- coding: utf-8 -*-
"""
Map ReEDS geographic regions and classes to Supply Curve points
"""
import logging
import numpy as np
import os
import pandas as pd
from warnings import warn

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
                 cap_bins=5, sort_bins_by='trans_cap_cost', pre_filter=None,
                 trg_by_region=False):
        """
        Parameters
        ----------
        rev_table : str | pandas.DataFrame
            reV supply curve or aggregation table,
            or path to file containing table
        resource_classes : str | pandas.DataFrame
            Resource classes, either provided in a .csv, .json or a DataFrame
            Allowable columns:
            - 'class' -> class labels to use
            - 'TRG_cap' -> TRG capacity bins to use to create TRG classes
            - any column in 'rev_table' -> Used for categorical bins
            - '*_min' and '*_max' where * is a numberical column in 'rev_table'
              -> used for range binning
            NOTE: 'TRG_cap' can only be combined with categorical bins
        region_map : str | pandas.DataFrame
            Mapping of supply curve points to region to create classes for
        cap_bins : int
            Number of equal capacity bins to create for each
            region-class
        sort_bins_by : str | list
            Column(s) to sort by before capacity binning
        pre_filter : dict | NoneType
            Column value pair(s) to filter on. If None don't filter
        trg_by_region : bool
            Groupby on region when computing TRGs
        """
        rev_table = self._parse_table(rev_table)
        if pre_filter is not None:
            for col, v in pre_filter.items():
                logger.debug('Subsetting reV table to {} in {}'
                             .format(v, col))
                mask = rev_table[col] == v
                rev_table = rev_table.loc[mask]

        rev_table = self._map_region(rev_table, region_map)
        rev_table = self._resource_classes(rev_table, resource_classes,
                                           trg_by_region=trg_by_region)
        self._rev_table = self._capacity_bins(rev_table, cap_bins,
                                              sort_bins_by=sort_bins_by)
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
    def _TRG_bins(rev_table, trg_bins, by_region=False):
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
        cap_breaks = np.concatenate(([0., ], cap_breaks),
                                    axis=0)
        labels = trg_bins.index.values

        cols = ['sc_gid', 'capacity', 'mean_lcoe', 'region']
        trg_classes = rev_table[cols].copy()
        if by_region:
            classes = []
            trg_classes['class'] = 1
            for _, df in trg_classes.groupby('region'):
                df = df.sort_values('mean_lcoe')
                cum_cap = df['capacity'].cumsum()
                df.loc[:, 'class'] = pd.cut(x=cum_cap, bins=cap_breaks,
                                            labels=labels)
                classes.append(df)

            trg_classes = pd.concat(classes)
        else:
            trg_classes = trg_classes.sort_values('mean_lcoe')
            cum_cap = trg_classes['capacity'].cumsum()
            trg_classes.loc[:, 'class'] = pd.cut(x=cum_cap, bins=cap_breaks,
                                                 labels=labels)

        rev_table = rev_table.merge(trg_classes[['sc_gid', 'class']],
                                    on='sc_gid', how='left')

        return rev_table

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

        Raises
        ------
        ValueError
            If categorical columns do not exist in rev_table
        """
        cat_cols = [c for c in trg_bins if c != 'TRG_cap']
        if cat_cols:
            missing = [c for c in cat_cols if c not in rev_table]
            if missing:
                msg = ("categorical column(s) supplied with 'TRG_cap' "
                       "are not valid columns of 'rev_table': {}"
                       .format(missing))
                logger.error(msg)
                raise ValueError(msg)
            else:
                msg = ("Additional columns were supplied with "
                       "'TRG_cap'! \n TRG bins will be computed for all "
                       "unique combinations of {}".format(cat_cols))
                logger.warning(msg)
                warn(msg)

            tables = []
            rev_groups = rev_table.groupby(cat_cols)
            for grp, bins in trg_bins.groupby(cat_cols):
                group_table = rev_groups.get_group(grp)
                tables.append(ReedsClassifier._TRG_bins(group_table, bins,
                                                        by_region=by_region))

            rev_table = pd.concat(tables).reset_index(drop=True)
        else:
            rev_table = ReedsClassifier._TRG_bins(rev_table, trg_bins,
                                                  by_region=by_region)

        return rev_table

    @staticmethod
    def _bin_classes(rev_table, class_bins):
        """
        Bin classes based on categorical or range bins

        Parameters
        ----------
        rev_table : pandas.DataFrame
            reV supply curve or aggregation table
        class_bins : pandas.DataFrame
            Class bins to use:
            - categorical: single value
            - range: *_min and *_max pair of values -> (min, max]

        Returns
        -------
        rev_table : pandas.DataFrame
            Updated table with TRG classes added

        Raises
        ------
        ValueError
            If range min and max are not supplied for range bins
        """
        range_cols = [c for c in class_bins if c.endswith(('min', 'max'))]

        if len(range_cols) % 2 != 0:
            msg = ("A '*_min' and a '*_max' value are neede for range bins! "
                   "Values provided: {}".format(range_cols))
            logger.error(msg)
            raise ValueError(msg)

        rev_cols = [c.rstrip('_min').rstrip('_max') for c in range_cols]
        rev_cols = list(set(rev_cols))
        for col in rev_cols:
            cols = ['{}_min'.format(col), '{}_max'.format(col)]
            class_bins[col] = list(class_bins[cols].values)

        class_bins = class_bins.drop(columns=range_cols)
        missing = [c for c in class_bins if c not in rev_table]
        if missing:
            msg = "Bin columns {} are not in 'rev_table'!".format(missing)
            logger.error(msg)
            raise ValueError(msg)

        rev_table['class'] = None
        for label, bins in class_bins.iterrows():
            mask = True
            for col, value in bins.iteritems():
                if isinstance(value, (list, np.ndarray)):
                    bin_mask = ((rev_table[col] > value[0])
                                & (rev_table[col] <= value[1]))
                else:
                    bin_mask = rev_table[col] == value

                mask *= bin_mask

            rev_table.loc[mask, 'class'] = label

        return rev_table

    @staticmethod
    def _resource_classes(rev_table, resource_classes, trg_by_region=False):
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
        resource_classes = ReedsClassifier._parse_table(resource_classes)
        if 'class' in resource_classes:
            resource_classes = resource_classes.set_index('class')

        if 'TRG_cap' in resource_classes:
            rev_table = ReedsClassifier._TRG_classes(rev_table,
                                                     resource_classes,
                                                     by_region=trg_by_region)
        else:
            rev_table = ReedsClassifier._bin_classes(rev_table,
                                                     resource_classes)

        return rev_table

    @staticmethod
    def _capacity_bins(rev_table, cap_bins, sort_bins_by='trans_cap_cost'):
        """
        Create equal capacity bins in each region-class sorted by given
        column(s)

        Parameters
        ----------
        rev_table : pandas.DataFrame
            reV supply curve or aggregation table
        cap_bins : int
            Number of equal capacity bins to create for each
            region-class
        sort_bins_by : str | list, optional
            Column(s) to sort by before capacity binning,
            by default 'trans_cap_cost'

        Returns
        -------
        rev_table : pandas.DataFrame
            Updated table with classes
        """
        if not isinstance(sort_bins_by, list):
            sort_bins_by = [sort_bins_by]

        cols = ['sc_gid', 'capacity', 'region', 'class'] + sort_bins_by
        capacity_bins = rev_table[cols].copy()

        bins = []
        capacity_bins['bin'] = 1
        labels = list(range(1, cap_bins + 1))
        for g, df in capacity_bins.groupby(['region', 'class']):
            df = df.sort_values(sort_bins_by)
            cum_cap = df['capacity'].cumsum()
            bin_labels = pd.cut(x=cum_cap, bins=cap_bins, labels=labels)
            unique_l = np.unique(bin_labels)
            if len(unique_l) < (cap_bins / 2):
                msg = ("In {}: only {} bins where filled: {}"
                       .format(g, len(unique_l), unique_l))
                warn(msg)
                logger.warning(msg)

            df.loc[:, 'bin'] = bin_labels
            bins.append(df)

        capacity_bins = pd.concat(bins)
        rev_table = rev_table.merge(capacity_bins[['sc_gid', 'bin']],
                                    on='sc_gid', how='left')

        return rev_table

    @classmethod
    def create(cls, rev_table, resource_classes, region_map='reeds_region',
               cap_bins=5, sort_bins_by='trans_cap_cost',
               pre_filter=None, trg_by_region=False):
        """
        Identify ReEDS regions and classes and dump and updated table

        Parameters
        ----------
        rev_table : str | pandas.DataFrame
            reV supply curve or aggregation table,
            or path to file containing table
        resource_classes : str | pandas.DataFrame
            Resource classes, either provided in a .csv, .json or a DataFrame
            Allowable columns:
            - 'class' -> class labels to use
            - 'TRG_cap' -> TRG capacity bins to use to create TRG classes
            - any column in 'rev_table' -> Used for categorical bins
            - '*_min' and '*_max' where * is a numberical column in 'rev_table'
              -> used for range binning
            NOTE: 'TRG_cap' can only be combined with categorical bins
        region_map : str | pandas.DataFrame
            Mapping of supply curve points to region to create classes for
        cap_bins : int
            Number of equal capacity bins to create for each
            region-class
        sort_bins_by : str | list, optional
            Column(s) to sort by before capacity binning,
            by default 'trans_cap_cost'
        pre_filter : dict | NoneType
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
                      cap_bins=cap_bins, sort_bins_by=sort_bins_by,
                      pre_filter=pre_filter, trg_by_region=trg_by_region)
        out = (classes.table, classes.table_slim, classes.aggregate_table,
               classes.aggregate_table_slim)

        return out
