# -*- coding: utf-8 -*-
"""
Pipeline between reV and RPM
"""
from concurrent.futures import as_completed
import logging
import os
import pandas as pd
import psutil
from warnings import warn

from rex.utilities.execution import SpawnProcessPool

from reVX.handlers.outputs import Outputs
from reVX.rpm.rpm_clusters import RPMClusters
from reVX.rpm.rpm_output import RPMOutput
from reVX.utilities.exceptions import RPMValueError, RPMRuntimeError

logger = logging.getLogger(__name__)


class RPMClusterManager:
    """
    RPM Cluster Manager:
    - Extracts gids for all RPM regions
    - Runs RPMClusters in parallel for all regions
    - Save results to disk
    """
    def __init__(self, cf_fpath, rpm_meta, rpm_region_col=None,
                 max_workers=None):
        """
        Parameters
        ----------
        cf_fpath : str
            Path to reV .h5 file containing desired capacity factor profiles
        rpm_meta : pandas.DataFrame | str
            DataFrame or path to .csv or .json containing the RPM meta data:
            (region, gid | gen_gid, clusters)
            - Regions of interest
            - # of clusters per region
            - cf or resource GIDs if region is not in default meta data
        rpm_region_col : str | Nonetype
            If not None, the meta-data field to map RPM regions to
        max_workers : int, optional
            Number of parallel workers. 1 will run serial, None will use all
            available., by default None
        """
        if rpm_region_col is not None:
            logger.info('Initializing RPM clustering on regional column "{}".'
                        .format(rpm_region_col))

        self._cf_h5 = cf_fpath
        self._rpm_regions = self._map_rpm_regions(rpm_meta,
                                                  region_col=rpm_region_col)

        if max_workers is None:
            max_workers = os.cpu_count()

        self.max_workers = max_workers

    @staticmethod
    def _parse_rpm_meta(rpm_meta):
        """
        Extract rpm meta and map it to the cf profile data

        Parameters
        ----------
        rpm_meta : pandas.DataFrame | str
            DataFrame or path to .csv or .json containing the RPM meta data:
            (region, gid | gen_gid, clusters)
            - Regions of interest
            - # of clusters per region
            - cf or resource GIDs if region is not in default meta data

        Returns
        -------
        rpm_meta : pandas.DataFrame
            DataFrame of RPM regional meta data (clusters and cf/resource GIDs)
        """
        if isinstance(rpm_meta, str):
            if rpm_meta.endswith('.csv'):
                rpm_meta = pd.read_csv(rpm_meta)
            elif rpm_meta.endswith('.json'):
                rpm_meta = pd.read_json(rpm_meta)
            else:
                raise RPMValueError("Cannot read RPM meta, "
                                    "file must be a '.csv' or '.json'")
        elif not isinstance(rpm_meta, pd.DataFrame):
            raise RPMValueError("RPM meta must be supplied as a pandas "
                                "DataFrame or as a .csv, or .json file")

        return rpm_meta

    def _map_rpm_regions(self, rpm_meta, region_col=None):
        """
        Map RPM meta to cf_profile gids

        Parameters
        ----------
        rpm_meta : pandas.DataFrame | str
            DataFrame or path to .csv or .json containing the RPM meta data:
            - Regions of interest
            - # of clusters per region
            - cf or resource GIDs if region is not in default meta data
        region_col : str | Nonetype
            If not None, the meta-data field to map RPM regions to

        Returns
        -------
        rpm_regions : dict
            Dictionary mapping rpm regions to cf GIDs and number of
            clusters
        """
        rpm_meta = self._parse_rpm_meta(rpm_meta)

        with Outputs(self._cf_h5, mode='r') as cfs:
            cf_meta = cfs.meta

        cf_meta.index.name = 'gen_gid'
        cf_meta = cf_meta.reset_index()

        rpm_regions = {}
        for region, region_df in rpm_meta.groupby('region'):
            region_map = {}
            if 'gid' in region_df:
                pos = cf_meta['gid'].isin(region_df['gid'].values)
                region_meta = cf_meta.loc[pos]
            elif 'gen_gid' in region_df:
                pos = cf_meta['gen_gid'].isin(region_df['gen_gid'].values)
                region_meta = cf_meta.loc[pos]
            elif region_col in cf_meta:
                pos = cf_meta[region_col] == region
                region_meta = cf_meta.loc[pos]
            else:
                raise RPMRuntimeError("Resource gids or a valid resource "
                                      "meta-data field must be supplied "
                                      "to map RPM regions")

            clusters = region_df['clusters'].unique()
            if len(clusters) > 1:
                raise RPMRuntimeError("Multiple values for 'clusters' "
                                      "were provided for region {}"
                                      .format(region))

            if region_meta['gen_gid'].empty:
                wmsg = ('Could not locate any generation in region "{}". '
                        'Region will be excluded.'
                        .format(region))
                warn(wmsg)
                logger.warning(wmsg)
            else:
                region_map['cluster_num'] = clusters[0]
                region_map['gen_gids'] = region_meta['gen_gid'].values
                region_map['gids'] = region_meta['gid'].values
                rpm_regions[region] = region_map

        return rpm_regions

    def _cluster(self, method='kmeans', method_kwargs=None,
                 dist_rank_filter=True, dist_rmse_kwargs=None,
                 contiguous_filter=True, contiguous_kwargs=None):
        """
        Cluster all RPM regions

        Parameters
        ----------
        method : str
            Method to use to cluster coefficients
        method_kwargs : dict
            Kwargs for running _cluster_coefficients
        dist_rank_filter : bool
            Run _optimize_dist_rank
        dist_rmse_kwargs : dict
            Kwargs for running _dist_rank_optimization
        contiguous_filter : bool
            Run _contiguous_filter
        contiguous_kwargs : dict
            Kwargs for _contiguous_filter
        """
        kwargs = {"method": method, "method_kwargs": method_kwargs,
                  "dist_rank_filter": dist_rank_filter,
                  "dist_rmse_kwargs": dist_rmse_kwargs,
                  "contiguous_filter": contiguous_filter,
                  "contiguous_kwargs": contiguous_kwargs}
        if self.max_workers > 1:
            future_to_region = {}
            loggers = [__name__, 'reVX.rpm.rpm_clusters', 'reVX']
            with SpawnProcessPool(max_workers=self.max_workers,
                                  loggers=loggers) as exe:
                for region, region_map in self._rpm_regions.items():
                    logger.info('Kicking off clustering for "{}".'
                                .format(region))
                    clusters = region_map['cluster_num']
                    gen_gids = region_map['gen_gids']

                    future = exe.submit(RPMClusters.cluster, self._cf_h5,
                                        gen_gids, clusters, **kwargs)
                    future_to_region[future] = region

                for i, future in enumerate(as_completed(future_to_region)):
                    mem = psutil.virtual_memory()
                    region = future_to_region[future]
                    logger.info('Finished clustering "{}", {} out of {}. '
                                'Memory usage is {:.2f} out of {:.2f} GB.'
                                .format(region, i + 1, len(future_to_region),
                                        mem.used / 1e9, mem.total / 1e9))
                    result = future.result()
                    self._rpm_regions[region].update({'clusters': result})

        else:
            for region, region_map in self._rpm_regions.items():
                logger.info('Kicking off clustering for "{}".'.format(region))
                clusters = region_map['cluster_num']
                gen_gids = region_map['gen_gids']
                result = RPMClusters.cluster(self._cf_h5, gen_gids, clusters,
                                             **kwargs)
                self._rpm_regions[region].update({'clusters': result})

    @staticmethod
    def _combine_region_clusters(rpm_regions):
        """
        Combine clusters for all rpm regions and create unique cluster ids

        Parameters
        ----------
        rpm_regions : dict
            Dictionary with RPM region info

        Returns
        -------
        rpm_clusters : pandas.DataFrame
            Single DataFrame with (region, gid, cluster_id, rank)
        """
        rpm_clusters = []
        for region, r_dict in rpm_regions.items():
            r_df = r_dict['clusters'].copy()
            ids = region + '-' + r_df.copy()['cluster_id'].astype(str).values
            r_df.loc[:, 'cluster_id'] = ids
            r_df['gid'] = r_dict['gids']
            rpm_clusters.append(r_df)

        rpm_clusters = pd.concat(rpm_clusters, sort=False)
        rpm_clusters = rpm_clusters.reset_index(drop=True)

        if 'geometry' in rpm_clusters:
            rpm_clusters = rpm_clusters.drop('geometry', axis=1)

        return rpm_clusters

    @classmethod
    def run_clusters(cls, cf_fpath, rpm_meta, out_dir, job_tag=None,
                     rpm_region_col=None, max_workers=True, **cluster_kwargs):
        """
        RPM Cluster Manager:
        - Extracts gen_gids for all RPM regions
        - Runs RPMClusters in parallel for all regions
        - Save results to disk

        Parameters
        ----------
        cf_fpath : str
            Path to reV .h5 file containing desired capacity factor profiles
        rpm_meta : pandas.DataFrame | str
            DataFrame or path to .csv or .json containing the RPM meta data:
            - Regions of interest
            - # of clusters per region
            - cf or resource GIDs if region is not in default meta data
        out_dir : str
            Directory to dump output files.
        job_tag : str | None
            Optional name tag to add to the output files.
            Format is "rpm_cluster_output_{tag}.csv".
        rpm_region_col : str | Nonetype
            If not None, the meta-data field to map RPM regions to
        max_workers : int, optional
            Number of parallel workers. 1 will run serial, None will use all
            available., by default None
        output_kwargs : dict | None
            Kwargs for the RPM outputs manager.
        **cluster_kwargs : dict
            RPMClusters kwargs
        """
        f_out = os.path.join(out_dir, 'rpm_clusters.csv')
        if job_tag is not None:
            f_out = f_out.replace('.csv', '_{}.csv'.format(job_tag))

        rpm = cls(cf_fpath, rpm_meta, rpm_region_col=rpm_region_col,
                  max_workers=max_workers)
        rpm._cluster(**cluster_kwargs)
        rpm_clusters = rpm._combine_region_clusters(rpm._rpm_regions)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        rpm_clusters.to_csv(f_out, index=False)

        return rpm_clusters

    @classmethod
    def run_clusters_and_profiles(cls, cf_fpath, rpm_meta, excl_fpath,
                                  excl_dict, techmap_dset, out_dir,
                                  job_tag=None, rpm_region_col=None,
                                  max_workers=True, output_kwargs=None,
                                  **cluster_kwargs):
        """
        RPM Cluster Manager:
        - Extracts gen_gids for all RPM regions
        - Runs RPMClusters in parallel for all regions
        - Save results to disk

        Parameters
        ----------
        cf_fpath : str
            Path to reV .h5 file containing desired capacity factor profiles
        rpm_meta : pandas.DataFrame | str
            DataFrame or path to .csv or .json containing the RPM meta data:
            - Regions of interest
            - # of clusters per region
            - cf or resource GIDs if region is not in default meta data
        excl_fpath : str | None
            Filepath to exclusions data (must match the techmap grid).
            None will not apply exclusions.
        excl_dict : dict | None
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
        techmap_dset : str
            Dataset name in the exclusions file containing the
            exclusions-to-resource mapping data.
        out_dir : str
            Directory to dump output files.
        job_tag : str | None
            Optional name tag to add to the output files.
            Format is "rpm_cluster_output_{tag}.csv".
        rpm_region_col : str | Nonetype
            If not None, the meta-data field to map RPM regions to
        max_workers : int, optional
            Number of parallel workers. 1 will run serial, None will use all
            available., by default None
        output_kwargs : dict | None
            Kwargs for the RPM outputs manager.
        **cluster_kwargs : dict
            RPMClusters kwargs
        """

        # intermediate job file
        f_cluster = os.path.join(out_dir, 'rpm_initial_clusters.csv')
        if job_tag is not None:
            f_cluster = f_cluster.replace('.csv', '_{}.csv'
                                          .format(job_tag))

        if not os.path.exists(f_cluster):
            rpm = cls(cf_fpath, rpm_meta, rpm_region_col=rpm_region_col,
                      max_workers=max_workers)
            rpm._cluster(**cluster_kwargs)
            rpm_clusters = rpm._combine_region_clusters(rpm._rpm_regions)

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            rpm_clusters.to_csv(f_cluster, index=False)

        else:
            logger.info('Importing initial cluster results from: {}'
                        .format(f_cluster))
            rpm_clusters = f_cluster
            rpm = None

        if output_kwargs is None:
            output_kwargs = {}

        RPMOutput.process_outputs(rpm_clusters, cf_fpath, excl_fpath,
                                  excl_dict, techmap_dset, out_dir,
                                  job_tag=job_tag, max_workers=max_workers,
                                  cluster_kwargs=cluster_kwargs,
                                  **output_kwargs)
        logger.info('reV-to-RPM processing is complete.')

        return rpm
