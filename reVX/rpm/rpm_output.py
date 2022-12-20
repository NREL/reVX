# -*- coding: utf-8 -*-
"""
RPM output handler.
"""
from concurrent.futures import as_completed
import logging
import numpy as np
import os
import pandas as pd
import psutil
from scipy.spatial import cKDTree
from scipy.ndimage import sum_labels
from warnings import warn

from reV.handlers.exclusions import ExclusionLayers
from reV.supply_curve.exclusions import ExclusionMask, ExclusionMaskFromDict
from reVX.handlers.outputs import Outputs
from reVX.rpm.rpm_clusters import RPMClusters
from reVX.utilities.exceptions import RPMRuntimeError, RPMTypeError
from reVX.utilities.utilities import log_versions
from rex.utilities.execution import SpawnProcessPool

logger = logging.getLogger(__name__)


class RepresentativeProfiles:
    """Methods to export representative generation profiles."""

    def __init__(self, clusters, cf_fpath, key=None, forecast_fpath=None):
        """
        Parameters
        ----------
        clusters : pd.DataFrame
            Single DataFrame with (gid, gen_gid, cluster_id, rank).
        cf_fpath : str
            reV generation output file.
        key : str | None
            Rank column to sort by to get the best ranked profile.
            None will use implicit logic to select the rank key.
        forecast_fpath : str
            reV generation output file for forecast data. If this is input,
            profiles will be taken from forecast fpath instead of fpath gen
            based on a NN mapping.
        """
        log_versions(logger)
        if key is not None:
            self.key = key
        elif 'rank_included_trg' in clusters:
            self.key = 'rank_included_trg'
        else:
            if 'rank_included' in clusters:
                self.key = 'rank_included'
            else:
                self.key = 'rank'

        if self.key not in clusters:
            raise KeyError('Could not find rank column "{}" in '
                           'cluster table. Cannot extract '
                           'representative profiles.'.format(self.key))

        logger.debug('Getting rep profiles based on column "{}".'
                     .format(key))

        self.clusters = clusters
        self._cf_fpath = cf_fpath
        self._forecast_fpath = forecast_fpath
        self._forecast_map = None

        if (self._forecast_fpath is not None
                and 'forecast_gid' not in self.clusters.columns):
            self.clusters = self.process_forecast_clusters(
                self.clusters, self._cf_fpath, self._forecast_fpath)

    @staticmethod
    def _make_forecast_nn_map(meta_cf, meta_forecast):
        """Make a mapping between the cf meta and the forecast meta.

        Parameters
        ----------
        meta_cf : pd.DataFrame
            Meta data for reV gen CF file (actuals).
        meta_forecast : pd.DataFrame
            Meta data for reV gen CF file (forecast).

        Returns
        -------
        i : np.ndarray
            1D array of forecast gid's with length equal to meta_cf.
        """
        labels = ['latitude', 'longitude']
        tree = cKDTree(meta_forecast[labels])  # pylint: disable=not-callable
        d, i, = tree.query(meta_cf[labels], k=1)
        logger.info('Mapping reV gen file to forecast gen file, '
                    'nearest neighbor min / mean / max: {} / {} / {}'
                    .format(d.min(), d.mean(), d.max()))
        return i

    @staticmethod
    def _add_forecast_gids(clusters, forecast_map, meta_forecast):
        """Add forecast_gid column in cluster table with forecast data.

        Parameters
        ----------
        clusters : pd.DataFrame
            Single DataFrame with (gid, gen_gid, cluster_id, rank).
        forecast_map : np.ndarray
            1D array of forecast gid's with length equal to meta_cf.
        meta_forecast : pd.DataFrame
            Meta data for reV gen CF file (forecast).

        Returns
        -------
        clusters : pd.DataFrame
            Single DataFrame with additional forecast columns.
        """
        clusters['forecast_gid'] = np.nan
        clusters['forecast_latitude'] = np.nan
        clusters['forecast_longitude'] = np.nan
        lats = meta_forecast['latitude']
        lons = meta_forecast['longitude']
        for i in clusters.index:
            gen_gid = clusters.at[i, 'gen_gid']
            forecast_gid = forecast_map[gen_gid]
            clusters.at[i, 'forecast_gid'] = forecast_gid
            clusters.at[i, 'forecast_latitude'] = lats[forecast_gid]
            clusters.at[i, 'forecast_longitude'] = lons[forecast_gid]

        return clusters

    @staticmethod
    def _get_rep_profile(clusters, cf_fpath, irp=0, fpath_out=None,
                         key='rank', forecast_fpath=None, cols=None):
        """Get a single representative profile timeseries dataframe.

        Parameters
        ----------
        clusters : pd.DataFrame
            Single DataFrame with (gid, gen_gid, cluster_id, rank).
        cf_fpath : str
            reV generation output file.
        irp : int
            Rank of profile to get. Zero is the most representative profile.
        fpath_out : str
            Optional filepath to export files to.
        key : str
            Rank column to sort by to get the best ranked profile.
        forecast_fpath : str
            reV generation output file for forecast data. If this is input,
            profiles will be taken from forecast fpath instead of fpath gen
            based on a NN mapping.
        cols : list | None
            Columns headers for the rep profiles. None will use whatever
            cluster_ids are in clusters.
        """
        if forecast_fpath is None:
            with Outputs(cf_fpath) as f:
                ti = f.time_index
        else:
            with Outputs(forecast_fpath) as f:
                ti = f.time_index

        if cols is None:
            cols = clusters.cluster_id.unique()

        profile_df = pd.DataFrame(index=ti, columns=cols)
        profile_df.index.name = 'time_index'

        for cid, df in clusters.groupby('cluster_id'):
            mask = ~df[key].isnull()
            if any(mask):
                df_ranked = df[mask].sort_values(by=key)
                if irp < len(df_ranked):
                    rep = df_ranked.iloc[irp, :]
                    res_gid = rep['gid']
                    gen_gid = rep['gen_gid']

                    if forecast_fpath is None:
                        logger.info('Representative profile i #{} from '
                                    'cluster id {} is from gen_gid {}, '
                                    'res_gid {}'
                                    .format(irp, cid, gen_gid, res_gid))

                        with Outputs(cf_fpath) as f:
                            meta_gid = f.get_meta_arr('gid')
                            gen_gid_arr = np.where(meta_gid == res_gid)[0]
                            if gen_gid_arr.size > 0:
                                gen_gid = gen_gid_arr[0]
                                profile_df.loc[:, cid] = f['cf_profile', :,
                                                           gen_gid]

                    else:
                        for_gid = rep['forecast_gid']
                        logger.info('Representative profile i #{} from '
                                    'cluster id {} is from gen_gid {}, '
                                    'forecast_gid {}.'
                                    .format(irp, cid, for_gid, gen_gid))

                        with Outputs(forecast_fpath) as f:
                            profile_df.loc[:, cid] = f['cf_profile', :,
                                                       for_gid]

        if fpath_out is not None:
            profile_df.to_csv(fpath_out)
            logger.info('Saved {}'.format(fpath_out))

    @classmethod
    def process_forecast_clusters(cls, clusters, cf_fpath, forecast_fpath):
        """Process the clusters dataframe with NN to forecast data.

        Parameters
        ----------
        clusters : pd.DataFrame
            Single DataFrame with (gid, gen_gid, cluster_id, rank).
        cf_fpath : str
            reV generation output file.
        forecast_fpath : str
            reV generation output file for forecast data. If this is input,
            profiles will be taken from forecast fpath instead of fpath gen
            based on a NN mapping.

        Returns
        -------
        clusters : pd.DataFrame
            Single DataFrame with additional forecast columns.
        """
        with Outputs(cf_fpath) as cf:
            meta_cf = cf.meta

        with Outputs(forecast_fpath) as forecast:
            meta_forecast = forecast.meta

        forecast_map = cls._make_forecast_nn_map(meta_cf, meta_forecast)
        clusters = cls._add_forecast_gids(clusters, forecast_map,
                                          meta_forecast)
        return clusters

    @classmethod
    def export_profiles(cls, n_profiles, clusters, cf_fpath, fn_pro,
                        out_dir, max_workers=1, key=None, forecast_fpath=None):
        """Export representative profile files.

        Parameters
        ----------
        n_profiles : int
            Number of profiles to export.
        clusters : pd.DataFrame
            RPM output clusters attribute.
        cf_fpath : str
            Filepath to reV generation results to get profiles from.
        fn_pro : str
            Filename for representative profile output.
        out_dir : str
            Directory to dump output files.
        key : str | None
            Column in clusters to sort ranks by. None will allow for
            default logic.
        forecast_fpath : str
            reV generation output file for forecast data. If this is input,
            profiles will be taken from forecast fpath instead of fpath gen
            based on a NN mapping.
        """
        if forecast_fpath is not None:
            clusters = cls.process_forecast_clusters(clusters, cf_fpath,
                                                     forecast_fpath)
            fn_fore = fn_pro.replace('.csv', '_meta.csv')
            clusters.to_csv(os.path.join(out_dir, fn_fore))

        if max_workers == 1:
            for irp in range(n_profiles):
                fni = fn_pro.replace('.csv', '_rank{}.csv'.format(irp))
                fpath_out_i = os.path.join(out_dir, fni)
                cls.export_single_profile(clusters, cf_fpath, irp=irp,
                                          fpath_out=fpath_out_i, key=key,
                                          forecast_fpath=forecast_fpath)
        else:
            loggers = [__name__, 'reVX']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                for irp in range(n_profiles):
                    fni = fn_pro.replace('.csv', '_rank{}.csv'.format(irp))
                    fpath_out_i = os.path.join(out_dir, fni)
                    exe.submit(cls.export_single_profile, clusters,
                               cf_fpath, irp=irp, fpath_out=fpath_out_i,
                               key=key, forecast_fpath=forecast_fpath)

    @classmethod
    def export_single_profile(cls, clusters, cf_fpath, irp=0, fpath_out=None,
                              key=None, forecast_fpath=None):
        """Get a single representative profile timeseries dataframe.

        Parameters
        ----------
        clusters : pd.DataFrame
            Single DataFrame with (gid, gen_gid, cluster_id, rank).
        cf_fpath : str
            reV generation output file.
        irp : int
            Rank of profile to get. Zero is the most representative profile.
        fpath_out : str
            Optional filepath to export files to.
        key : str | None
            Rank column to sort by to get the best ranked profile.
            None will use implicit logic to select the rank key.
        forecast_fpath : str
            reV generation output file for forecast data. If this is input,
            profiles will be taken from forecast fpath instead of fpath gen
            based on a NN mapping.
        """
        rp = cls(clusters, cf_fpath, key=key, forecast_fpath=forecast_fpath)
        cols = clusters.cluster_id.unique()

        if rp.key == 'rank_included_trg':
            for itrg, df in rp.clusters.groupby('trg'):
                if fpath_out is not None:
                    fpath_out_trg = fpath_out.replace('.csv', '_trg{}.csv'
                                                      .format(itrg))

                rp._get_rep_profile(df, cf_fpath, irp=irp,
                                    fpath_out=fpath_out_trg, key=rp.key,
                                    forecast_fpath=rp._forecast_fpath,
                                    cols=cols)
        else:
            rp._get_rep_profile(rp.clusters, cf_fpath, irp=irp,
                                fpath_out=fpath_out, key=rp.key,
                                forecast_fpath=rp._forecast_fpath,
                                cols=cols)


class RPMOutput:
    """Framework to format and process RPM clustering results."""

    def __init__(self, rpm_clusters, cf_fpath, excl_fpath, excl_dict,
                 techmap_dset, excl_area=None, include_threshold=0.001,
                 n_profiles=1, rerank=True, cluster_kwargs=None,
                 max_workers=None, trg_bins=None, trg_dset='lcoe_fcr',
                 pre_extract_inclusions=False, target_crs=None):
        """
        Parameters
        ----------
        rpm_clusters : pd.DataFrame | str
            Single DataFrame with (gid, gen_gid, cluster_id, rank),
            or str to file.
        cf_fpath : str
            Path to reV .h5 file containing desired capacity factor profiles
        excl_fpath : str | None
            Filepath to exclusions data (must match the techmap grid).
            None will not apply exclusions.
        excl_dict : dict | None
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
        techmap_dset : str
            Dataset name in the exclusions file containing the
            exclusions-to-resource mapping data.
        excl_area : float | None
            Area in km2 of one exclusion pixel. None will calculate the
            exclusion pixel area from the exclusion projection profile.
        include_threshold : float
            Inclusion threshold. Resource pixels included more than this
            threshold will be considered in the representative profiles.
            Set to zero to find representative profile on all resource, not
            just included.
        n_profiles : int
            Number of representative profiles to output.
        rerank : bool
            Flag to rerank representative generation profiles after removing
            excluded generation pixels.
        cluster_kwargs : dict
            RPMClusters kwargs
        max_workers : int, optional
            Number of parallel workers. 1 will run serial, None will use all
            available., by default None
        trg_bins : str | list | None
            TRG bins as an ordered list of bin edge values or string to a csv
            containing a single column with bin edge values. None will ignore
            trgs.
        trg_dset : str
            Dataset associated with TRG bins that can be found in the cf_fpath
            file.
        pre_extract_inclusions : bool
            Flag to pre-extract the inclusion mask using excl_fpath and
            excl_dict. This is advantageous if the excl_dict is highly complex
            and if you're processing a lot of points. Default is False.
        target_crs : str
            Target coordinate reference system, e.g. "EPSG:4326". Note that
            everything is in EPSG:4326/WGS84 by default (raw lat/lon with lon=0
            at Greenwich).
        """

        logger.info('Initializing RPM output processing...')

        self._clusters = self._parse_cluster_arg(rpm_clusters)

        self._excl_fpath = excl_fpath
        self._excl_dict = excl_dict

        self._techmap_dset = techmap_dset
        self._cf_fpath = cf_fpath
        self.excl_area = excl_area
        self.include_threshold = include_threshold
        self.n_profiles = n_profiles
        self.rerank = rerank
        self.target_crs = target_crs

        if self.excl_area is None and self._excl_fpath is not None:
            with ExclusionLayers(self._excl_fpath) as excl:
                self.excl_area = excl.pixel_area

        if max_workers is None:
            max_workers = os.cpu_count()

        self.max_workers = max_workers

        if cluster_kwargs is None:
            self.cluster_kwargs = {}
        else:
            self.cluster_kwargs = cluster_kwargs

        self.trg_bins = trg_bins
        self.trg_dset = trg_dset
        self.trg_labels = None
        if isinstance(self.trg_bins, str):
            self.trg_bins = pd.read_csv(self.trg_bins)
            msg = 'trg csv can only have one column'
            assert len(self.trg_bins.columns.values) == 1, msg
            col = self.trg_bins.columns.values[0]
            self.trg_bins = self.trg_bins[col].values.tolist()

        if self.trg_bins is not None:
            # bins must be in monotonic ascending order for pd.cut but labels
            # should be ordered however the input is received
            self.trg_labels = [i + 1 for i in range(len(self.trg_bins) - 1)]
            incr = (np.diff(self.trg_bins) > 0).all()
            if not incr:
                self.trg_bins = self.trg_bins[::-1]
                self.trg_labels.reverse()

        self._excl_lat = None
        self._excl_lon = None
        self._full_lat_slice = None
        self._full_lon_slice = None
        self._init_lat_lon()

        self._inclusion_mask = None
        self._techmap_data = None
        if pre_extract_inclusions:
            logger.info('Pre-extracting exclusions mask, '
                        'this could take a while...')
            with ExclusionMaskFromDict(self._excl_fpath) as excl:
                self._techmap_data = excl.excl_h5[self._techmap_dset]
                self._techmap_data = self._techmap_data.astype(np.int32)

            self._inclusion_mask = \
                ExclusionMaskFromDict.extract_inclusion_mask(
                    self._excl_fpath, self._techmap_dset,
                    excl_dict=self._excl_dict)

    @classmethod
    def _parse_cluster_arg(cls, rpm_clusters):
        """Parse dataframe from cluster input arg.

        Parameters
        ----------
        rpm_clusters : pd.DataFrame | str
            Single DataFrame with (gid, gen_gid, cluster_id, rank),
            or str to file.

        Returns
        -------
        clusters : pd.DataFrame
            Single DataFrame with (gid, gen_gid, cluster_id, rank,
            latitude, longitude)
        """

        clusters = None
        if isinstance(rpm_clusters, pd.DataFrame):
            clusters = rpm_clusters

        elif isinstance(rpm_clusters, str):
            if rpm_clusters.endswith('.csv'):
                clusters = pd.read_csv(rpm_clusters)
            elif rpm_clusters.endswith('.json'):
                clusters = pd.read_json(rpm_clusters)

        if clusters is None:
            raise RPMTypeError('Expected a DataFrame or str but received {}'
                               .format(type(rpm_clusters)))

        cls._check_cluster_cols(clusters)

        return clusters

    @staticmethod
    def _check_cluster_cols(df, required=('gen_gid', 'gid', 'latitude',
                                          'longitude', 'cluster_id', 'rank')):
        """Check for required columns in the rpm cluster dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Single DataFrame with columns to check
        """

        missing = []
        for c in required:
            if c not in df:
                missing.append(c)

        if any(missing):
            raise RPMRuntimeError('Missing the following columns in RPM '
                                  'clusters input df: {}'.format(missing))

    def _init_lat_lon(self):
        """Initialize the lat/lon arrays and reduce their size."""

        if self._excl_fpath is not None:

            self._full_lat_slice, self._full_lon_slice = \
                self._get_lat_lon_slices(cluster_id=None)

            logger.debug('Initial lat/lon shape is {} and {} and '
                         'range is {} - {} and {} - {}'
                         .format(self.excl_lat.shape, self.excl_lon.shape,
                                 self.excl_lat.min(), self._excl_lat.max(),
                                 self.excl_lon.min(), self._excl_lon.max()))
            self._excl_lat = self._excl_lat[self._full_lat_slice,
                                            self._full_lon_slice]
            self._excl_lon = self._excl_lon[self._full_lat_slice,
                                            self._full_lon_slice]
            logger.debug('Reduced lat/lon shape is {} and {} and '
                         'range is {} - {} and {} - {}'
                         .format(self.excl_lat.shape, self.excl_lon.shape,
                                 self.excl_lat.min(), self._excl_lat.max(),
                                 self.excl_lon.min(), self._excl_lon.max()))

    @staticmethod
    def _get_tm_data(excl, techmap_dset, lat_slice, lon_slice):
        """Get the techmap data.

        Parameters
        ----------
        excl : ExclusionMask | ExclusionMaskFromDict
            Pre-initialized exclusions mask object.
        techmap_dset : str
            Dataset name in the exclusions file containing the
            exclusions-to-resource mapping data.
        lat_slice : slice
            The latitude (row) slice to extract from the exclusions or
            techmap 2D datasets.
        lon_slice : slice
            The longitude (col) slice to extract from the exclusions or
            techmap 2D datasets.

        Returns
        -------
        techmap : np.ndarray
            Techmap data mapping exclusions grid to resource gid (flattened).
        """
        if isinstance(excl, (ExclusionMask, ExclusionMaskFromDict)):
            techmap = excl.excl_h5[techmap_dset, lat_slice, lon_slice]
        else:
            e = 'Cannot recognize exclusion type: {}'.format(type(excl))
            logger.error(e)
            raise TypeError(e)

        techmap = techmap.astype(np.int32).flatten()

        return techmap

    @staticmethod
    def _get_incl_mask(excl, lat_slice, lon_slice):
        """Get the exclusions data from a geotiff file.

        Parameters
        ----------
        excl : ExclusionMask | ExclusionMaskFromDict
            Pre-initialized exclusions mask object.
        lat_slice : slice
            The latitude (row) slice to extract from the exclusions or
            techmap 2D datasets.
        lon_slice : slice
            The longitude (col) slice to extract from the exclusions or
            techmap 2D datasets.

        Returns
        -------
        incl_data : np.ndarray
            Inclusions data mask flattened and normalized
            from 0 to 1 (1 is incld).
        """

        if isinstance(excl, (ExclusionMask, ExclusionMaskFromDict)):
            incl_data = excl[lat_slice, lon_slice]
        else:
            e = 'Cannot recognize exclusion type: {}'.format(type(excl))
            logger.error(e)
            raise TypeError(e)

        # infer exclusions that are scaled percentages from 0 to 100
        if incl_data.max() > 1:
            incl_data = incl_data.astype(np.float32)
            incl_data /= 100

        return incl_data.flatten()

    def _get_lat_lon_slices(self, cluster_id=None, margin=0.1):
        """Get the slice args to locate exclusion/techmap data of interest.

        Parameters
        ----------
        cluster_id : str | None
            Single cluster ID of interest or None for full region.
        margin : float
            Extra margin around the cluster lat/lon box.

        Returns
        -------
        lat_slice : slice
            The latitude (row) slice to extract from the exclusions or
            techmap 2D datasets.
        lon_slice : slice
            The longitude (col) slice to extract from the exclusions or
            techmap 2D datasets.
        """

        box = self._get_coord_box(cluster_id)

        mask = ((self.excl_lat > np.min(box['latitude']) - margin)
                & (self.excl_lat < np.max(box['latitude']) + margin)
                & (self.excl_lon > np.min(box['longitude']) - margin)
                & (self.excl_lon < np.max(box['longitude']) + margin))

        if not mask.any():
            msg = ('Lat Lon box retrieval failed for cluster "{}". The '
                   'exclusion lat min/max is {:.2f}/{:.2f} and lon min/max '
                   'is {:.2f}/{:.2f} while the cluster box is: {}'
                   .format(cluster_id,
                           self.excl_lat.min(), self.excl_lat.max(),
                           self.excl_lon.min(), self.excl_lon.max(),
                           box))
            logger.error(msg)
            raise RPMRuntimeError(msg)

        lat_locs, lon_locs = np.where(mask)

        if self._full_lat_slice is None and self._full_lon_slice is None:
            lat_slice = slice(np.min(lat_locs), 1 + np.max(lat_locs))
            lon_slice = slice(np.min(lon_locs), 1 + np.max(lon_locs))
        else:
            lat_slice = slice(
                self._full_lat_slice.start + np.min(lat_locs),
                1 + self._full_lat_slice.start + np.max(lat_locs))
            lon_slice = slice(
                self._full_lon_slice.start + np.min(lon_locs),
                1 + self._full_lon_slice.start + np.max(lon_locs))

        return lat_slice, lon_slice

    def _get_all_lat_lon_slices(self, margin=0.1, free_mem=True):
        """Get the slice args for all clusters.

        Parameters
        ----------
        margin : float
            Extra margin around the cluster lat/lon box.
        free_mem : bool
            Flag to free lat/lon arrays from memory to clear space for later
            exclusion processing.

        Returns
        -------
        slices : dict
            Dictionary of tuples - (lat_slice, lon_slice) slices
            keyed by cluster id.
        """

        slices = {}
        for cid in self._clusters['cluster_id'].unique():
            slices[cid] = self._get_lat_lon_slices(cluster_id=cid,
                                                   margin=margin)

        if free_mem:
            # free up memory
            self._excl_lat = None
            self._excl_lon = None
            self._full_lat_slice = None
            self._full_lon_slice = None

        return slices

    def _get_coord_box(self, cluster_id=None):
        """Get the RPM cluster latitude/longitude range.

        Parameters
        ----------
        cluster_id : str | None
            Single cluster ID of interest or None for all clusters in
            self._clusters.

        Returns
        -------
        coord_box : dict
            Bounding box of the cluster or region:
                {'latitude': (lat_min, lat_max),
                 'longitude': (lon_min, lon_max)}
        """

        if cluster_id is not None:
            mask = (self._clusters['cluster_id'] == cluster_id)
        else:
            mask = len(self._clusters) * [True]

        lat_range = (self._clusters.loc[mask, 'latitude'].min(),
                     self._clusters.loc[mask, 'latitude'].max())
        lon_range = (self._clusters.loc[mask, 'longitude'].min(),
                     self._clusters.loc[mask, 'longitude'].max())
        box = {'latitude': lat_range, 'longitude': lon_range}

        return box

    @property
    def excl_lat(self):
        """Get the full 2D array of latitudes of the exclusion grid.

        Returns
        -------
        _excl_lat : np.ndarray
            2D array representing the latitudes at each exclusion grid cell
        """

        if self._excl_lat is None and self._excl_fpath is not None:
            with Outputs(self._excl_fpath) as f:
                logger.debug('Importing Latitude data from techmap...')
                self._excl_lat = f['latitude']

        return self._excl_lat

    @property
    def excl_lon(self):
        """Get the full 2D array of longitudes of the exclusion grid.

        Returns
        -------
        _excl_lon : np.ndarray
            2D array representing the latitudes at each exclusion grid cell
        """

        if self._excl_lon is None and self._excl_fpath is not None:
            with Outputs(self._excl_fpath) as f:
                logger.debug('Importing Longitude data from techmap...')
                self._excl_lon = f['longitude']

        return self._excl_lon

    @classmethod
    def _single_excl(cls, cluster_id, clusters, excl_fpath, excl_dict,
                     techmap_dset, lat_slice, lon_slice, techmap_subset=None,
                     incl_mask_subset=None):
        """Calculate the exclusions for each resource GID in a cluster.

        Parameters
        ----------
        cluster_id : str
            Single cluster ID of interest.
        clusters : pandas.DataFrame
            Single DataFrame with (gid, gen_gid, cluster_id, rank)
        excl_fpath : str | None
            Filepath to exclusions data (must match the techmap grid).
            None will not apply exclusions.
        excl_dict : dict | None
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
        techmap_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        lat_slice : slice
            The latitude (row) slice to extract from the exclusions or
            techmap 2D datasets.
        lon_slice : slice
            The longitude (col) slice to extract from the exclusions or
            techmap 2D datasets.
        techmap_subset : None | np.ndarray
            Optional techmap data subset (just for this lat_slice, lon_slice)
            in the case that inlcusions were pre-extracted. This must be a
            flattened 1D array (original will have been a 2D exclusion raster).
        incl_mask_subset : None | np.ndarray
            Optional inclusion mask subset (just for this lat_slice, lon_slice)
            in the case that inclusions were pre-extracted. This must be a
            flattened 1D array (original will have been a 2D exclusion raster).

        Returns
        -------
        inclusions : np.ndarray
            1D array of inclusions fraction corresponding to the indexed
            cluster provided by cluster_id.
        n_inclusions : np.ndarray
            1D array of number of (at least partially) included pixels
            corresponding to each gid in cluster_id (sum of inclusion fractions
            rounded up).
        n_points : np.ndarray
            1D array of the total number of techmap pixels corresponding to
            each gid in cluster_id.
        """

        mask = (clusters['cluster_id'] == cluster_id)
        res_gids = clusters.loc[mask, 'gid'].values.astype(np.int32)
        locs = np.where(mask)[0]
        inclusions = np.zeros((len(locs), ), dtype=np.float32)
        n_inclusions = np.zeros((len(locs), ), dtype=np.float32)
        n_points = np.zeros((len(locs), ), dtype=np.uint16)

        if techmap_subset is None or incl_mask_subset is None:
            with ExclusionMaskFromDict(
                    excl_fpath, layers_dict=excl_dict) as excl:
                techmap_subset = cls._get_tm_data(excl, techmap_dset,
                                                  lat_slice, lon_slice)
                incl_mask_subset = cls._get_incl_mask(excl, lat_slice,
                                                      lon_slice)
        else:
            msg = ('Techmap subset must be 1D but received shape {}'
                   .format(techmap_subset.shape))
            assert len(techmap_subset.shape) == 1, msg
            assert len(incl_mask_subset.shape) == 1, msg

        inclusions = sum_labels(input=incl_mask_subset, labels=techmap_subset,
                                index=res_gids)
        n_inclusions = sum_labels(input=np.ceil(incl_mask_subset),
                                  labels=techmap_subset, index=res_gids)
        n_points = sum_labels(input=np.ones_like(incl_mask_subset),
                              labels=techmap_subset, index=res_gids)

        n_inclusions[(n_points == 0)] = np.nan
        inclusions /= n_points

        return inclusions, n_inclusions, n_points

    def _apply_exclusions_parallel(self, unique_clusters, static_clusters,
                                   slices):
        """Calculate exclusions for clusters in parallel

        Parameters
        ----------
        unique_clusters : np.ndarray
            1D array of unique cluster id's.
        static_clusters : pd.DataFrame
            Static (non-changing deepcopy) version of self._clusters.
        slices : dict
            Dictionary of tuples - (lat_slice, lon_slice)
            slices keyed by cluster id.
        """

        futures = {}
        loggers = [__name__, 'reVX']
        with SpawnProcessPool(max_workers=self.max_workers,
                              loggers=loggers) as exe:
            for i, cid in enumerate(unique_clusters):

                lat_s, lon_s = slices[cid]
                techmap_subset = None
                incl_mask_subset = None
                if (self._techmap_data is not None
                        and self._inclusion_mask is not None):
                    techmap_subset = self._techmap_data[lat_s, lon_s].flatten()
                    incl_mask_subset = self._inclusion_mask[lat_s, lon_s]
                    incl_mask_subset = incl_mask_subset.flatten()

                future = exe.submit(self._single_excl, cid, static_clusters,
                                    self._excl_fpath, self._excl_dict,
                                    self._techmap_dset,
                                    lat_s, lon_s,
                                    techmap_subset=techmap_subset,
                                    incl_mask_subset=incl_mask_subset)
                futures[future] = cid
                logger.debug('Kicked off exclusions for cluster "{}", {} out '
                             'of {}.'.format(cid, i + 1, len(unique_clusters)))

            for i, future in enumerate(as_completed(futures)):
                cid = futures[future]
                mem = psutil.virtual_memory()
                logger.info('Finished exclusions for cluster "{}", {} out '
                            'of {} futures. '
                            'Memory usage is {:.2f} out of {:.2f} GB.'
                            .format(cid, i + 1, len(futures),
                                    mem.used / 1e9, mem.total / 1e9))
                incl, n_incl, n_pix = future.result()
                mask = (self._clusters['cluster_id'] == cid)

                self._clusters.loc[mask, 'included_frac'] = incl
                self._clusters.loc[mask, 'included_area_km2'] = \
                    n_incl * self.excl_area
                self._clusters.loc[mask, 'n_excl_pixels'] = n_pix

    def _apply_exclusions_serial(self, unique_clusters, static_clusters,
                                 slices):
        """Calculate exclusions for clusters in serial

        Parameters
        ----------
        unique_clusters : np.ndarray
            1D array of unique cluster id's.
        static_clusters : pd.DataFrame
            Static (non-changing deepcopy) version of self._clusters.
        slices : dict
            Dictionary of tuples - (lat_slice, lon_slice)
            slices keyed by cluster id.
        """

        for i, cid in enumerate(unique_clusters):

            lat_s, lon_s = slices[cid]
            techmap_subset = None
            incl_mask_subset = None
            if (self._techmap_data is not None
                    and self._inclusion_mask is not None):
                techmap_subset = self._techmap_data[lat_s, lon_s].flatten()
                incl_mask_subset = self._inclusion_mask[lat_s, lon_s].flatten()

            incl, n_incl, n_pix = self._single_excl(
                cid, static_clusters, self._excl_fpath, self._excl_dict,
                self._techmap_dset, lat_s, lon_s,
                techmap_subset=techmap_subset,
                incl_mask_subset=incl_mask_subset)

            mem = psutil.virtual_memory()
            logger.info('Finished exclusions for cluster "{}", {} out '
                        'of {} futures. '
                        'Memory usage is {:.2f} out of {:.2f} GB.'
                        .format(cid, i + 1, len(unique_clusters),
                                mem.used / 1e9, mem.total / 1e9))

            mask = (self._clusters['cluster_id'] == cid)

            self._clusters.loc[mask, 'included_frac'] = incl
            self._clusters.loc[mask, 'included_area_km2'] = \
                n_incl * self.excl_area
            self._clusters.loc[mask, 'n_excl_pixels'] = n_pix

    def apply_exclusions(self):
        """Calculate exclusions for clusters, adding data to self._clusters.

        Returns
        -------
        self._clusters : pd.DataFrame
            self._clusters with new columns for exclusions data.
        """

        logger.info('Working on applying exclusions with {} workers...'
                    .format(self.max_workers))

        unique_clusters = self._clusters['cluster_id'].unique()
        static_clusters = self._clusters.copy()
        self._clusters['included_frac'] = 0.0
        self._clusters['included_area_km2'] = 0.0
        self._clusters['n_excl_pixels'] = 0

        slices = self._get_all_lat_lon_slices()

        if self.max_workers > 1:
            self._apply_exclusions_parallel(unique_clusters, static_clusters,
                                            slices)
        else:
            self._apply_exclusions_serial(unique_clusters, static_clusters,
                                          slices)

        logger.info('Finished applying exclusions.')

        if self.rerank:
            self.run_rerank(groupby='cluster_id', rank_col='rank_included')

        return self._clusters

    def apply_trgs(self):
        """Apply TRG's if requested."""

        with Outputs(self._cf_fpath) as f:
            dsets = f.datasets

        if self.trg_bins is not None and self.trg_dset not in dsets:
            wmsg = ('TRGs requested but "{}" not in cf file: {}'
                    .format(self.trg_dset, self._cf_fpath))
            warn(wmsg)
            logger.warning(wmsg)

        if self.trg_bins is not None and self.trg_dset in dsets:
            gen_gid = sorted(list(self._clusters['gen_gid'].values))
            with Outputs(self._cf_fpath) as f:
                trg_data = f[self.trg_dset, gen_gid]

            trg_df = pd.DataFrame({'gen_gid': gen_gid,
                                   self.trg_dset: trg_data})

            label = 'trg_bin_{}'.format(self.trg_dset)
            trg_df[label] = pd.cut(x=trg_df[self.trg_dset], bins=self.trg_bins)
            trg_df['trg'] = pd.cut(x=trg_df[self.trg_dset], bins=self.trg_bins,
                                   labels=self.trg_labels)

            self._clusters = pd.merge(self._clusters, trg_df, on='gen_gid',
                                      how='left', validate='1:1')

            self.run_rerank(groupby=['cluster_id', 'trg'],
                            rank_col='rank_included_trg')

    def _run_rerank_parallel(self, groupby='cluster_id',
                             rank_col='rank_included'):
        """Re-rank rep profiles for included resource in generic groups.

        Parameters
        ----------
        groupby : str | list
            One or more columns in self._clusters to groupby and rank profiles
            within each group.
        rank_col : str
            Column to add to self._clusters with new rankings.
        """

        futures = {}
        loggers = [__name__, 'reVX.rpm.rpm_clusters', 'reVX']
        with SpawnProcessPool(max_workers=self.max_workers,
                              loggers=loggers) as exe:
            for _, df in self._clusters.groupby(groupby):
                if 'included_frac' in df:
                    mask = (df['included_frac'] >= self.include_threshold)
                else:
                    mask = [True] * len(df)

                if any(mask):
                    gen_gid = df.loc[mask, 'gen_gid']
                    self.cluster_kwargs['dist_rank_filter'] = False
                    self.cluster_kwargs['contiguous_filter'] = False
                    future = exe.submit(RPMClusters.cluster, self._cf_fpath,
                                        gen_gid, 1, **self.cluster_kwargs)
                    futures[future] = gen_gid

            if futures:
                logger.info('Re-ranking representative profiles "{}" using '
                            'groupby: {}'.format(rank_col, groupby))
                self._clusters[rank_col] = np.nan

            for i, future in enumerate(as_completed(futures)):
                gen_gid = futures[future]
                mem = psutil.virtual_memory()
                logger.info('Finished re-ranking {} out of {}. '
                            'Memory usage is {:.2f} out of {:.2f} GB.'
                            .format(i, len(futures),
                                    mem.used / 1e9, mem.total / 1e9))
                new = future.result()
                mask = self._clusters['gen_gid'].isin(gen_gid)
                self._clusters.loc[mask, rank_col] = new['rank'].values

    def _run_rerank_serial(self, groupby='cluster_id',
                           rank_col='rank_included'):
        """Re-rank rep profiles for included resource in generic groups.

        Parameters
        ----------
        groupby : str | list
            One or more columns in self._clusters to groupby and rank profiles
            within each group.
        rank_col : str
            Column to add to self._clusters with new rankings.
        """
        init = False
        for _, df in self._clusters.groupby(groupby):

            if 'included_frac' in df:
                mask = (df['included_frac'] >= self.include_threshold)
            else:
                mask = [True] * len(df)

            if any(mask):
                if not init:
                    self._clusters[rank_col] = np.nan
                    logger.info('Re-ranking representative profiles "{}" '
                                'using groupby: {}'.format(rank_col, groupby))
                    init = True

                gen_gid = df.loc[mask, 'gen_gid']
                self.cluster_kwargs['dist_rank_filter'] = False
                self.cluster_kwargs['contiguous_filter'] = False
                new = RPMClusters.cluster(self._cf_fpath, gen_gid, 1,
                                          **self.cluster_kwargs)

                mask = self._clusters['gen_gid'].isin(gen_gid)
                self._clusters.loc[mask, rank_col] = new['rank'].values

    def run_rerank(self, groupby='cluster_id', rank_col='rank_included'):
        """Re-rank rep profiles for included resource in generic groups.

        Parameters
        ----------
        groupby : str | list
            One or more columns in self._clusters to groupby and rank profiles
            within each group.
        rank_col : str
            Column to add to self._clusters with new rankings.
        """
        if self.max_workers > 1:
            self._run_rerank_parallel(groupby=groupby, rank_col=rank_col)
        else:
            self._run_rerank_serial(groupby=groupby, rank_col=rank_col)

    @property
    def cluster_summary(self):
        """Summary dataframe with cluster_id primary key.

        Returns
        -------
        s : pd.DataFrame
            Summary dataframe with a row for each cluster id.
        """

        if ('included_frac' not in self._clusters
                and self._excl_fpath is not None
                and self._excl_dict is not None):
            raise RPMRuntimeError('Exclusions must be applied before '
                                  'representative profiles can be determined.')

        ind = self._clusters.cluster_id.unique()
        cols = ['latitude',
                'longitude',
                'n_gen_gids',
                'included_frac',
                'included_area_km2']
        s = pd.DataFrame(index=ind, columns=cols)
        s.index.name = 'cluster_id'

        for i, df in self._clusters.groupby('cluster_id'):
            s.loc[i, 'latitude'] = df['latitude'].mean()
            s.loc[i, 'longitude'] = df['longitude'].mean()
            s.loc[i, 'n_gen_gids'] = len(df)

            if 'included_frac' in df:
                s.loc[i, 'included_frac'] = df['included_frac'].mean()
                s.loc[i, 'included_area_km2'] = df['included_area_km2'].sum()

        return s

    def make_shape_file(self, fpath_shp):
        """Make shape file containing all clusters.

        Parameters
        ----------
        fpath_shp : str
            Filepath to write shape_file to.
        """

        labels = ['cluster_id', 'latitude', 'longitude']
        RPMClusters.generate_shapefile(self._clusters[labels], fpath_shp,
                                       target_crs=self.target_crs)

    @staticmethod
    def _get_fout_names(job_tag):
        """Get a set of output filenames.

        Parameters
        ----------
        job_tag : str | None
            Optional name tag to add to the csvs being saved.
            Format is "rpm_cluster_output_{tag}.csv".

        Returns
        -------
        fn_out : str
            Filename for full cluster output.
        fn_pro : str
            Filename for representative profile output.
        fn_sum : str
            Filename for summary output.
        fn_shp : str
            Filename for shapefile output.
        """

        fn_out = 'rpm_cluster_outputs.csv'
        fn_pro = 'rpm_rep_profiles.csv'
        fn_sum = 'rpm_cluster_summary.csv'
        fn_shp = 'rpm_cluster_shapes.shp'

        if job_tag is not None:
            fn_out = fn_out.replace('.csv', '_{}.csv'.format(job_tag))
            fn_pro = fn_pro.replace('.csv', '_{}.csv'.format(job_tag))
            fn_sum = fn_sum.replace('.csv', '_{}.csv'.format(job_tag))
            fn_shp = fn_shp.replace('.shp', '_{}.shp'.format(job_tag))

        return fn_out, fn_pro, fn_sum, fn_shp

    def export_all(self, out_dir, job_tag=None):
        """Run RPM output algorithms and write to CSV's.

        Parameters
        ----------
        out_dir : str
            Directory to dump output files.
        job_tag : str | None
            Optional name tag to add to the csvs being saved.
            Format is "rpm_cluster_output_{tag}.csv".
        """

        fn_out, fn_pro, fn_sum, fn_shp = self._get_fout_names(job_tag)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if ('included_frac' not in self._clusters
                and self._excl_fpath is not None
                and self._excl_dict is not None):
            self.apply_exclusions()

        self.apply_trgs()

        RepresentativeProfiles.export_profiles(
            self.n_profiles, self._clusters, self._cf_fpath, fn_pro, out_dir,
            max_workers=self.max_workers, key=None)

        self.cluster_summary.to_csv(os.path.join(out_dir, fn_sum))
        logger.info('Saved {}'.format(fn_sum))

        self._clusters.to_csv(os.path.join(out_dir, fn_out), index=False)
        logger.info('Saved {}'.format(fn_out))

        self.make_shape_file(os.path.join(out_dir, fn_shp))
        logger.info('Saved {}'.format(fn_shp))

    @classmethod
    def extract_profiles(cls, rpm_clusters, cf_fpath, out_dir, n_profiles=1,
                         job_tag=None, max_workers=None, key=None,
                         forecast_fpath=None):
        """Use pre-formatted RPM cluster outputs to generate profile outputs.

        Parameters
        ----------
        rpm_clusters : pd.DataFrame | str
            Single DataFrame with (gid, gen_gid, cluster_id, rank),
            or str to file.
        cf_fpath : str
            reV generation output file.
        out_dir : str
            Directory to dump output files.
        n_profiles : int
            Number of representative profiles to output.
        job_tag : str | None
            Optional name tag to add to the output files.
            Format is "rpm_cluster_output_{tag}.csv".
        max_workers : int, optional
            Number of parallel workers. 1 will run serial, None will use all
            available., by default None
        key : str | None
            Column in clusters to sort ranks by. None will allow for
            default logic.
        forecast_fpath : str
            reV generation output file for forecast data. If this is input,
            profiles will be taken from forecast fpath instead of fpath gen
            based on a NN mapping.
        """

        rpmo = cls(rpm_clusters, cf_fpath, None, None, None,
                   n_profiles=n_profiles, max_workers=max_workers)

        _, fn_pro, _, _ = rpmo._get_fout_names(job_tag)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        RepresentativeProfiles.export_profiles(
            rpmo.n_profiles, rpmo._clusters, rpmo._cf_fpath, fn_pro, out_dir,
            max_workers=rpmo.max_workers, key=key,
            forecast_fpath=forecast_fpath)

        logger.info('Finished extracting extra representative profiles!')

    @classmethod
    def process_outputs(cls, rpm_clusters, cf_fpath, excl_fpath,
                        excl_dict, techmap_dset, out_dir, job_tag=None,
                        max_workers=None, cluster_kwargs=None,
                        excl_area=None, include_threshold=0.001,
                        n_profiles=1, rerank=True,
                        trg_bins=None, trg_dset='lcoe_fcr',
                        pre_extract_inclusions=False, target_crs=None):
        """Perform output processing on clusters and write results to disk.

        Parameters
        ----------
        rpm_clusters : pd.DataFrame | str
            Single DataFrame with (gid, gen_gid, cluster_id, rank),
            or str to file.
        cf_fpath : str
            Path to reV .h5 file containing desired capacity factor profiles
        excl_fpath : str | None
            Filepath to exclusions data (must match the techmap grid).
            None will not apply exclusions.
        excl_dict : dict | None
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
        techmap_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        out_dir : str
            Directory to dump output files.
        job_tag : str | None
            Optional name tag to add to the output files.
            Format is "rpm_cluster_output_{tag}.csv".
        max_workers : int, optional
            Number of parallel workers. 1 will run serial, None will use all
            available., by default None
        excl_area : float | None
            Area in km2 of one exclusion pixel. None will calculate the
            exclusion pixel area from the exclusion projection profile.
        include_threshold : float
            Inclusion threshold. Resource pixels included more than this
            threshold will be considered in the representative profiles.
            Set to zero to find representative profile on all resource, not
            just included.
        n_profiles : int
            Number of representative profiles to output.
        rerank : bool
            Flag to rerank representative generation profiles after removing
            excluded generation pixels.
        trg_bins : str | list | None
            TRG bins as an ordered list of bin edge values or string to a csv
            containing a single column with bin edge values. None will ignore
            trgs.
        trg_dset : str
            Dataset associated with TRG bins that can be found in the cf_fpath
            file.
        pre_extract_inclusions : bool
            Flag to pre-extract the inclusion mask using excl_fpath and
            excl_dict. This is advantageous if the excl_dict is highly complex
            and if you're processing a lot of points. Default is False.
        target_crs : str
            Target coordinate reference system, e.g. "EPSG:4326". Note that
            everything is in EPSG:4326/WGS84 by default (raw lat/lon with lon=0
            at Greenwich).
        """

        rpmo = cls(rpm_clusters, cf_fpath, excl_fpath, excl_dict,
                   techmap_dset, cluster_kwargs=cluster_kwargs,
                   max_workers=max_workers, excl_area=excl_area,
                   include_threshold=include_threshold, n_profiles=n_profiles,
                   rerank=rerank, trg_bins=trg_bins, trg_dset=trg_dset,
                   pre_extract_inclusions=pre_extract_inclusions,
                   target_crs=target_crs)
        rpmo.export_all(out_dir, job_tag=job_tag)
