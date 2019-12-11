# -*- coding: utf-8 -*-
"""
Extract ReEDS timeslices from rep-profiles
"""
import concurrent.futures as cf
import json
import logging
import numpy as np
import os
import pandas as pd
from scipy.stats import mode
from reV.handlers.resource import Resource

from reVX.reeds.reeds_classification import ReedsClassifier
from reVX.utilities.exceptions import ReedsValueError, ReedsRuntimeError

logger = logging.getLogger(__name__)


class ReedsTimeslices:
    """
    Create ReEDS timeslices from region-bin-class groups and representative
    profiles
    """
    def __init__(self, profiles, timeslice_map, rev_table=None,
                 reg_cols=('region', 'class')):
        """
        Parameters
        ----------
        profiles : str
            Path to .h5 file containing profiles (representative or cf)
        timeslice_map : str | pandas.DataFrame
            Path to timeslice mapping file or DataFrame with mapping
        rev_table : str | pandas.DataFrame
            reV supply curve or aggregation table, or path to file containing
            table. Only required if timeslice calc is based on all cf_profiles
            instead of representative profiles.
        reg_cols : tuple
            Label(s) for a categorical region column(s) to create timeslices
            for
        """
        self._profiles, self._meta, self._time_index = \
            self._check_profiles(profiles, rev_table=rev_table,
                                 reg_cols=reg_cols)
        self._timeslice_groups = self._parse_timeslices(timeslice_map,
                                                        self._time_index)
        if rev_table is not None:
            self._cf_profiles = True
        else:
            self._cf_profiles = False

    def __repr__(self):
        if self._cf_profiles:
            profiles = 'cf profiles'
        else:
            profiles = 'representative profiles'

        msg = ("{} with {} timeslices and {} region-class groups computed "
               "from {}".format(self.__class__.__name__, len(self.timeslices),
                                len(self.regions), profiles))
        return msg

    @property
    def profiles(self):
        """
        Representative profiles Multi-index DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        return self._profiles

    @property
    def meta(self):
        """
        Meta data for "region" groups

        Returns
        -------
        pandas.DataFrame
        """
        return self._meta

    @property
    def regions(self):
        """
        Unique "regions" defaulted to region-class pairs

        Returns
        -------
        list
        """
        return self._meta.index

    @property
    def timeslices(self):
        """
        Timeslice keys

        Returns
        -------
        list
        """
        return sorted(list(self._timeslice_groups.keys()))

    @property
    def profile_datetimes(self):
        """
        Datetime index for representative profiles

        Returns
        -------
        pandas.DatetimeIndex
        """
        return self._time_index

    @staticmethod
    def _extract_meta_and_timeindex(profiles):
        """
        Extract meta data, and time index from the profiles .h5 file

        Parameters
        ----------
        profiles : str
            Path to .h5 file containing profiles (representative or cf)

        Returns
        -------
        meta : pandas.DataFrame
            Meta data table corresponding to profiles
        time_index : pandas.DatetimeIndex
            Datetime Index for profiles
        """
        with Resource(profiles) as f:
            meta = f.meta
            if 'rev_summary' in f.dsets:
                table = f._get_meta('rev_summary', slice(None))
                reg_cols = list(meta.columns.drop(['rep_gen_gid',
                                                   'rep_res_gid']))
                tz = table.groupby(reg_cols)
                tz = tz['timezone'].apply(lambda x: mode(x).mode[0])
                meta = meta.merge(tz.reset_index(), on=reg_cols)
                meta = meta.set_index(reg_cols)

            if 'timezone' not in meta:
                msg = ('Meta data must contain timzone to allow conversion '
                       'to local time!')
                logger.error(msg)
                raise ReedsRuntimeError(msg)

            time_index = f.time_index

        return meta, time_index

    @staticmethod
    def _unpack_list(col):
        """
        Unpack lists stored in given column

        Parameters
        ----------
        col : pandas.DataFrame | pandas.Series
            Column containing lists to unpack

        Returns
        ------
        ndarray
            Vector of values contained in lists
        """
        if isinstance(col.iloc[0], str):
            col = col.apply(json.loads)

        return np.concatenate(col.values)

    @staticmethod
    def _unpack_gen_gids(group_df):
        """
        Unpack gen gids and counts lists

        Parameters
        -----------
        group_df : pandas.DataFrame
            Group DataFrame to unpack gen gids and counts for

        Returns
        -------
        pandas.DataFrame
            DataFrame of unique gen gids and their associated counts
        """
        gen_df = {
            'gid': ReedsTimeslices._unpack_list(group_df['gen_gids']),
            'gid_count': ReedsTimeslices._unpack_list(group_df['gid_counts'])}

        return pd.DataFrame(gen_df).groupby('gid').sum()

    @staticmethod
    def _add_reg_cols(meta, rev_table, reg_cols=('region', 'class')):
        """
        Add reg_cols to meta from rev_table

        Parameters
        ----------
        meta : pandas.DataFrame
            Meta data table corresponding to profiles, must include timezone
        rev_table : None | str | pandas.DataFrame
            reV supply curve or aggregation table,
            or path to file containing table
        reg_cols : tuple
            Label(s) for a categorical region column(s) to create timeslices
            for

        Returns
        -------
        meta : pandas.DataFrame
            Updated meta with reg_cols added to sites to be extract
            timeslices from
        """
        rev_table = ReedsClassifier._parse_table(rev_table)
        rev_table = rev_table.groupby(list(reg_cols))
        rev_table = rev_table.apply(ReedsTimeslices._unpack_gen_gids)

        meta = meta.merge(rev_table.reset_index(), on='gid')
        meta = meta.set_index(list(reg_cols))

        return meta

    @staticmethod
    def _check_profiles(profiles, rev_table=None,
                        reg_cols=('region', 'class')):
        """
        Check profiles to ensure all needed data is available.
        Extract meta and time_index

        Parameters
        ----------
        profiles : str
            Path to .h5 file containing profiles (representative or cf)
        rev_table : None | str | pandas.DataFrame
            reV supply curve or aggregation table,
            or path to file containing table
        reg_cols : tuple
            Label(s) for a categorical region column(s) to create timeslices
            for

        Returns
        -------
        profiles : str
            Path to .h5 file containing profiles (representative or cf)
        meta : pandas.DataFrame
            Meta data table corresponding to profiles, must include timezone
        time_index : pandas.DatetimeIndex
            Datetime Index for rep profiles
        """
        if not profiles.endswith('.h5'):
            msg = ("Cannot parse {} must be a .h5 file!"
                   .format(profiles))
            logger.error(msg)
            raise ReedsValueError(msg)

        if not os.path.isfile(profiles):
            msg = "{} is not a valid file path!".format(profiles)
            logger.error(msg)
            raise ReedsValueError(msg)

        meta, time_index = \
            ReedsTimeslices._extract_meta_and_timeindex(profiles)

        if rev_table is not None:
            meta = ReedsTimeslices._add_reg_cols(meta, rev_table,
                                                 reg_cols=reg_cols)

        logger.info('Profile data check complete.')

        return profiles, meta, time_index

    @staticmethod
    def _parse_timeslices(timeslice_map, time_index):
        """
        Extract timeslice mapping

        Parameters
        ----------
        timeslice_map : str | pandas.DataFrame
            Path to timeslice mapping file or DataFrame with mapping
        time_index : pandas.DatetimeIndex
            Datetime Index for profiles, must be supplied with rep_profiles
            dictionary

        Returns
        ----------
        timeslice_map : pandas.GroupBy
            Mapping of each timeslice to profiles time_index
        """
        if isinstance(timeslice_map, str):
            timeslice_map = pd.read_csv(timeslice_map)
        elif not isinstance(timeslice_map, pd.DataFrame):
            msg = ('Cannot parse timeslice mapping from : {}'
                   .format(type(timeslice_map)))
            logger.error(msg)
            raise ReedsValueError(msg)

        index_col = [c for c in timeslice_map.columns
                     if 'datetime' in c.lower()]
        if not index_col:
            msg = ('Timeslice mapping does not contain a "datetime" column!')
            logger.error(msg)
            raise ReedsRuntimeError(msg)

        if len(index_col) > 1:
            msg = ('Multiple possible "datetime" columns found!')
            logger.error(msg)
            raise ReedsRuntimeError(msg)

        timeslice_map = timeslice_map.set_index(index_col[0])
        timeslice_map.index = pd.to_datetime(timeslice_map.index)

        mask = timeslice_map.index.isin(time_index)
        if not mask.all():
            msg = ("{} timesteps in the timeslice mapping do not exist "
                   "in the representative profiles datetime index:"
                   "\n{}".format((~mask).sum(), timeslice_map.index[~mask]))
            logger.error(msg)
            raise ReedsRuntimeError(msg)

        cols = list(timeslice_map.columns)
        if len(cols) == 1:
            cols = cols[0]

        logger.info('Extracted timeslice map.')

        return timeslice_map.groupby(cols)

    @staticmethod
    def _roll_array(arr, shifts):
        """
        Roll array with unique shifts for each column
        This converts timeseries to local time

        Parameters
        ----------
        arr : ndarray
            Input timeseries array of form (time, sites)
        shifts : ndarray | list
            Vector of shifts from UTC to local time

        Returns
        -------
        local_arr : ndarray
            Array shifted to local time
        """
        if arr.shape[1] != len(shifts):
            msg = ('Number of timezone shifts ({}) does not match number of '
                   'sites ({})'.format(len(shifts), arr.shape[1]))
            logger.error(msg)
            raise ReedsValueError(msg)

        local_arr = np.empty(arr.shape, dtype=arr.dtype)
        for i, s in enumerate(shifts):
            local_arr[:, i] = np.roll(arr[:, i], s)

        return local_arr

    @staticmethod
    def _extract_rep_profiles(profiles_h5, meta):
        """
        Extract representative profiles and combine them with meta data
        and time_index to create a single dataframe to extract stats from

        Parameters
        ----------
        profiles_h5 : str
            Path to .h5 file containing representative profiles
        meta : pandas.DataFrame
            Meta data table for representative profiles

        Returns
        -------
        profiles_df : pandas.DataFrame
            Multi-index DataFrame of profiles for each 'region'
        """
        profiles = {}
        with Resource(profiles_h5) as f:
            time_index = f.time_index
            for ds in f.dsets:
                if 'rep_profiles' in ds:
                    k = int(ds.split('_')[-1])
                    profiles[k] = f[ds]

        cols = [str([int(i) for i in c]) for c in meta.index]
        tz = meta['timezone'].values
        tz *= len(time_index) // 8760
        temp = pd.DataFrame(columns=cols, index=time_index)
        profiles_df = []
        for k, arr in profiles.items():
            df = temp.copy()
            df.loc[:] = ReedsTimeslices._roll_array(arr, tz)
            df.columns = pd.MultiIndex.from_product([[k], df.columns])
            profiles_df.append(df.swaplevel(axis=1))

        profiles_df = pd.concat(profiles_df, axis=1).sort_index(axis=1,
                                                                level=0)

        return profiles_df

    @staticmethod
    def _rep_tslice_stats(ts_profiles):
        """
        Compute means and standard deviations and correlation coefficients
        for each timeslice from representative profiles

        Parameters
        ----------
        ts_profiles : pandas.DataFrame
            Timeseries profiles to compute stats for

        Returns
        -------
        means : pandas.Series
            CF means for each "region"
        stdevs : pandas.Series
            Standard deviations of CF for each "region"
        coeffs : pandas.DataFrame
            Pearson's correlation coefficients between all combinations
            of regions
        """
        cols = sorted(list(set(c[0] for c in ts_profiles.columns)))
        n = len(cols)
        data = ts_profiles.loc[:, (slice(None), 0)][cols].values

        if not np.all(data):
            coeffs = np.ones((n, n))
        else:
            coeffs = np.corrcoef(data, data, rowvar=False)

        coeffs = pd.DataFrame(coeffs[:n, :n], columns=cols, index=cols)
        coeffs = coeffs.fillna(1)

        means = ts_profiles.stack().mean()
        stdevs = ts_profiles.stack().std()

        return means, stdevs, coeffs

    @staticmethod
    def _rep_profile_stats(profiles_h5, meta, timeslice_groups,
                           max_workers=None):
        """
        Compute means and standard divations for each timeslice from
        representative profiles
        Parameters
        ----------
        profiles_h5 : str
            Path to .h5 file containing representative profiles
        meta : pandas.DataFrame
            Meta data table for representative profiles
        timeslice_groups : pandas.GroupBy
            Mapping of each timeslice to profiles time_index
        max_workers : int
            Number of workers to use for parallel computation of means
            and stdevs when using cf profiles.
            1 means run in serial
            None means use all available CPUs

        Returns
        -------
        means : pandas.DataFrame
            Mean CF for each region and timeslice
        stdevs : pandas.DataFrame
            Standard deviation in CF for each region and timeslice
        corr_coeffs : dict
            Correlation matrices for each timeslice
        """

        if max_workers is None:
            max_workers = os.cpu_count()

        profiles = ReedsTimeslices._extract_rep_profiles(profiles_h5, meta)
        means = []
        stdevs = []
        corr_coeffs = {}
        logger.info('Computing representative profile timeslice stats for '
                    '{} timeslice groups.'.format(len(timeslice_groups)))
        logger.info('Computing timeslice stats with max_workers: {}'
                    .format(max_workers))

        if max_workers > 1:
            with cf.ProcessPoolExecutor(max_workers=max_workers) as exe:
                futures = {}
                for s, slice_map in timeslice_groups:
                    tslice = profiles.loc[slice_map.index]
                    future = exe.submit(ReedsTimeslices._rep_tslice_stats,
                                        tslice)
                    futures[future] = s

                for i, future in enumerate(cf.as_completed(futures)):
                    s = futures[future]
                    mean, stdev, coeffs = future.result()

                    corr_coeffs[s] = coeffs
                    mean.name = s
                    means.append(mean)

                    stdev.name = s
                    stdevs.append(stdev)

                    logger.info('Completed {} out of {} representative '
                                'profile timeslice stats futures.'
                                .format(i + 1, len(futures)))

        else:
            for i, (s, slice_map) in enumerate(timeslice_groups):
                tslice = profiles.loc[slice_map.index]
                mean, stdev, coeffs = ReedsTimeslices._rep_tslice_stats(tslice)

                corr_coeffs[s] = coeffs
                mean.name = s
                means.append(mean)

                stdev.name = s
                stdevs.append(stdev)

                logger.info('Completed {} out of {} representative '
                            'profile timeslice stats.'
                            .format(i + 1, len(timeslice_groups)))

        means = pd.concat(means, axis=1).T
        means.index.name = 'timeslice'
        stdevs = pd.concat(stdevs, axis=1).T
        stdevs.index.name = 'timeslice'

        return means, stdevs, corr_coeffs

    @staticmethod
    def _cf_group_stats(profiles_h5, region_meta, timeslice_groups):
        """
        Compute means and standard deviations for each region

        Parameters
        ----------
        profiles_h5 : str
            Path to .h5 file containing profiles (representative or cf)
        region_meta : pandas.DataFrame
            Meta data table for subset of cf profiles in specific region
        timeslice_groups : pandas.GroupBy
            Mapping of each timeslice to profiles time_index

        Returns
        -------
        means : pandas.Series
            Mean CF for timeslices in specific region
        stdevs : pandas.Series
            Standard deviation in CF for timeslices in specific region
        """
        gids = region_meta['gid'].values
        tz = region_meta['timezone'].values
        counts = region_meta['gid_count'].values
        with Resource(profiles_h5) as f:
            time_index = f.time_index
            tz *= len(time_index) // 8760
            profiles = f['cf_profile', :, gids] * counts / np.sum(counts)

        profiles = pd.DataFrame(ReedsTimeslices._roll_array(profiles, tz),
                                index=time_index)
        means = {}
        stdevs = {}
        for s, slice_map in timeslice_groups:
            tslice = profiles.loc[slice_map.index]
            means[s] = tslice.stack().mean()
            stdevs[s] = tslice.stack().std()

        return pd.Series(means), pd.Series(stdevs)

    @staticmethod
    def _cf_profile_stats(profiles_h5, meta, timeslices, max_workers=None):
        """
        Compute timeslice mean and standard deviation from cf profiles

        Parameters
        ----------
        profiles_h5 : str
            Path to .h5 file containing profiles (representative or cf)
        meta : pandas.DataFrame
            Meta data table for cf profiles
        timeslices : pandas.DataFrame
            Mapping between timeslices of interest and profile timesteps

        Returns
        -------
        means : pandas.DataFrame
            Mean CF for each region and timeslice
        stdevs : pandas.DataFrame
            Standard deviation in CF for each region and timeslice
        """
        reg_cols = meta.index.names
        meta = meta.reset_index()

        if max_workers is None:
            max_workers = os.cpu_count()

        logger.info('Computing timeslice stats with max_workers: {}'
                    .format(max_workers))

        if max_workers > 1:
            with cf.ProcessPoolExecutor(max_workers=max_workers) as exe:
                futures = {}
                for group, df in meta.groupby(reg_cols):
                    future = exe.submit(ReedsTimeslices._cf_group_stats,
                                        profiles_h5, df, timeslices)
                    futures[future] = group

                logger.debug('Submitted {} futures.'.format(len(futures)))

                means = []
                stdevs = []
                for i, future in enumerate(cf.as_completed(futures)):
                    logger.info('Timeslice future {} out of {} completed.'
                                .format(i + 1, len(futures)))
                    m, s = future.result()
                    group = futures[future]
                    m.name = str(group)
                    s.name = str(group)
                    means.append(m)
                    stdevs.append(s)
        else:
            means = []
            stdevs = []
            for group, df in meta.groupby(reg_cols):
                m, s = ReedsTimeslices._cf_group_stats(profiles_h5, df,
                                                       timeslices)
                m.name = str(list(group))
                s.name = str(list(group))
                means.append(m)
                stdevs.append(s)

        means = pd.concat(means, axis=1)
        means.index.name = 'timeslice'
        stdevs = pd.concat(stdevs, axis=1)
        stdevs.index.name = 'timeslice'

        return means, stdevs

    @staticmethod
    def _flatten_timeslices(table, value_name, reg_cols):
        """
        Flatten timeslice table into legacy ReEDS Format

        Parameters
        ----------
        table : pandas.DataFrame
            Input means or stdev table to flatten
        value_name : str
            Name of values in table
        reg_cols : list
            List of names for "region" classifiers, default is region, class

        Returns
        -------
        out : pandas.DataFrame
            Flattened table
        """
        out = []
        for r, region in table.iteritems():
            if not isinstance(r, str):
                r = json.loads(r)

            region.name = value_name
            region = region.to_frame()

            for n, v in zip(reg_cols, r):
                region[n] = v

            out.append(region)

        return pd.concat(out).reset_index()

    @staticmethod
    def _create_correlation_table(corr_coeffs, reg_cols):
        """
        Flatten and combine correlation matrixes for all timeslices

        Parameters
        ----------
        coeffs : dict
            Correlation matrices for each timeslice
        reg_cols : list
            List of names for "region" classifiers, default is region, class

        Returns
        -------
        out : pandas.DataFrame
            Flattened table of correlation coefficients for all timeslices
        """

        logger.info('Creating legacy timeslice correlation table.')
        out = []
        for k, v in corr_coeffs.items():

            ctable = pd.DataFrame({'coefficient': v.values.flatten()})

            second_cols = []
            for i, rc in enumerate(reg_cols):
                r_label_0 = rc
                r_label_1 = rc + '2'
                second_cols.append(r_label_1)

                r0 = [json.loads(c)[i] if isinstance(c, str) else c[i]
                      for c in v.index]
                r1 = [json.loads(c)[i] if isinstance(c, str) else c[i]
                      for c in v.columns]

                r0 = [[r0[i]] * len(r1) for i in range(len(r1))]
                r0 = [item for sublist in r0 for item in sublist]
                r1 *= len(v.index)

                ctable[r_label_0] = r0
                ctable[r_label_1] = r1

            ctable['timeslice'] = k
            corder = ['coefficient'] + reg_cols + second_cols + ['timeslice']
            ctable = ctable[corder]

            out.append(ctable)

        sort_cols = (reg_cols + ["{}2".format(c) for c in reg_cols]
                     + ['timeslice', ])
        out = pd.concat(out).sort_values(sort_cols).reset_index(drop=True)
        logger.info('Legacy timeslice correlation table complete.')
        return out

    def _to_legacy_format(self, means, stdevs, coeffs=None):
        """
        Convert outputs to legacy format

        Parameters
        ----------
        means : pandas.DataFrame
            Mean CF for each region and timeslice
        stdevs : pandas.DataFrame
            Standard deviation in CF for each region and timeslice
        coeffs : dict
            Correlation matrices for each timeslice, optional

        Returns
        -------
        stats : pandas.DataFrame
            Flattened timeslice table with means and sigma
        coeffs : pandas.DataFrame
            Flattened correlation
        """
        logger.info('Performing legacy formatting operation.')
        reg_cols = list(self._meta.index.names)
        means = self._flatten_timeslices(means, 'cfmean', reg_cols)
        stdevs = self._flatten_timeslices(stdevs, 'cfsigma', reg_cols)
        merge_cols = reg_cols + ['timeslice', ]
        stats = means.merge(stdevs, on=merge_cols)
        sort_cols = reg_cols + ['timeslice', ]
        stats = stats.sort_values(sort_cols).reset_index(drop=True)
        logger.info('Legacy formatting complete.')

        if coeffs is not None:
            coeffs = self._create_correlation_table(coeffs, reg_cols)

        return stats, coeffs

    def compute_stats(self, max_workers=None, legacy_format=True):
        """
        Compute the mean and stdev CF for each timeslice for each "region"
        Compute the correlation coefficients between "regions" from
        timeslice means

        Parameters
        ----------
        max_workers : int
            Number of workers to use for parallel computation of means
            and stdevs when using cf profiles.
            1 means run in serial
            None means use all available CPUs
        legacy_format : bool
            Format outputs into ReEDS legacy format

        Returns
        -------
        stats : pandas.DataFrame | tuple
            In legacy format: a DataFrame containing means and stdevs
            (sigma), else a tuple of (means, stdevs) DataFrames
        corr_coeffs : pandas.DataFrame | dict | NoneType
            In legacy format: a DataFrame of correlation coefficients for
            all pairs of "regions" for each timeslice, else:
            a dictionary of correlation matrices for each timeslice.
            None for CF profiles due to memory constraints
        """
        if self._cf_profiles:
            means, stdevs = self._cf_profile_stats(self._profiles, self._meta,
                                                   self._timeslice_groups,
                                                   max_workers=max_workers)
            coeffs = None
        else:
            means, stdevs, coeffs = \
                self._rep_profile_stats(self._profiles, self._meta,
                                        self._timeslice_groups,
                                        max_workers=max_workers)

        logger.info('Finished timeslice stats computation.')

        if legacy_format:
            stats, coeffs = self._to_legacy_format(means, stdevs, coeffs)
        else:
            stats = (means, stdevs)

        return stats, coeffs

    @classmethod
    def run(cls, profiles, timeslice_map, rev_table=None,
            reg_cols=('region', 'class'), max_workers=None,
            legacy_format=True):
        """
        Compute means and standar deviations for each region and timeslice
        from given representative profiles

        Parameters
        ----------
        profiles : str
            Path to .h5 file containing profiles (representative or cf)
        timeslice_map : str | pandas.DataFrame
            Path to timeslice mapping file or DataFrame with mapping
        rev_table : str | pandas.DataFrame
            reV supply curve or aggregation table, or path to file containing
            table. Only required if timeslice calc is based on all cf_profiles
            instead of representative profiles.
        reg_cols : tuple
            Label(s) for a categorical region column(s) to create timeslices
            for
        max_workers : int
            Number of workers to use for parallel computation of means
            and stdevs when using cf profiles.
            1 means run in serial
            None means use all available CPUs
        legacy_format : bool
            Format outputs into ReEDS legacy format

        Returns
        -------
        stats : pandas.DataFrame | tuple
            In legacy format: a DataFrame containing means and stdevs
            (sigma), else a tuple of (means, stdevs) DataFrames
        corr_coeffs : pandas.DataFrame | dict | NoneType
            In legacy format: a DataFrame of correlation coefficients for
            all pairs of "regions" for each timeslice, else:
            a dictionary of correlation matrices for each timeslice.
            None for CF profiles due to memory constraints
        """
        ts = cls(profiles, timeslice_map, rev_table=rev_table,
                 reg_cols=reg_cols)
        stats, coeffs = ts.compute_stats(max_workers=max_workers,
                                         legacy_format=legacy_format)

        return stats, coeffs
