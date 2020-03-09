# -*- coding: utf-8 -*-
"""
RPM Clustering Module
"""
from copy import deepcopy
import geopandas as gpd
import logging
import numpy as np
import pywt
from scipy.spatial import cKDTree
from shapely.geometry import Point

from reVX.handlers.outputs import Outputs
from reVX.utilities.cluster_methods import ClusteringMethods

logger = logging.getLogger(__name__)


class RPMClusters:
    """
    Base class for RPM clusters

    Examples
    --------
    >>> from reV import Resource
    >>>
    >>> fname = '$TESTDATADIR/reV_gen/gen_pv_2012.h5'
    >>> with Resource(fname) as res:
    >>>     gen_gids = f.meta.index.values
    >>>
    >>> clusters = RPMClusters(fname, gen_gids, n_clusters=6)
    >>> clusters._cluster(**kwargs)
    >>> clusters.meta
            gen_gid   latitude  longitude  cluster_id   geometry
    0         0  41.290001 -71.860001           0  POINT (-71.86000 41.29000)
    1         1  41.290001 -71.820000           0  POINT (-71.82000 41.29000)
    2         2  41.250000 -71.820000           4  POINT (-71.82000 41.25000)
    3         3  41.330002 -71.820000           0  POINT (-71.82000 41.33000)
    4         4  41.369999 -71.820000           0  POINT (-71.82000 41.37000)
    ..      ...        ...        ...         ...                         ...
    95       95  41.250000 -71.660004           4  POINT (-71.66000 41.25000)
    96       96  41.889999 -71.660004           5  POINT (-71.66000 41.89000)
    97       97  41.450001 -71.660004           3  POINT (-71.66000 41.45000)
    98       98  41.610001 -71.660004           1  POINT (-71.66000 41.61000)
    99       99  41.410000 -71.660004           3  POINT (-71.66000 41.41000)

    Generate Shape File of Cluster

    >>> RPMClusters._generate_shapefile(clusters.meta, fpath='./test.shp')
    """
    def __init__(self, cf_fpath, gen_gids, n_clusters):
        """
        Parameters
        ----------
        cf_fpath : str
            Path to reV .h5 files containing desired capacity factor profiles
        gen_gids : list | ndarray
            List or vector of gen_gids to cluster on
        n_clusters : int
            Number of clusters to identify
        """
        self._meta, self._coefficients = self._parse_data(cf_fpath,
                                                          gen_gids)
        self._n_clusters = n_clusters

    @property
    def coefficients(self):
        """
        Returns
        -------
        _coefficients : ndarray
            Array of wavelet coefficients for each gen_gid
        """
        return self._coefficients

    @property
    def meta(self):
        """
        Returns
        -------
        _meta : pandas.DataFrame
            DataFrame of meta data:
            - gen_gid
            - latitude
            - longitude
            - cluster_id
            - rank
        """
        return self._meta

    @property
    def n_clusters(self):
        """
        Returns
        -------
        _n_clusters : int
            Number of clusters
        """
        return self._n_clusters

    @property
    def cluster_coefficients(self):
        """
        Returns
        -------
        cluster_coeffs : ndarray
            Representative coefficients for each cluster
        """
        cluster_coeffs = None
        if 'cluster_id' in self._meta:
            cluster_coeffs = []
            for _, cdf in self._meta.groupby('cluster_id'):
                idx = cdf.index.values
                cluster_coeffs.append(self.coefficients[idx].mean(axis=0))

            cluster_coeffs = np.array(cluster_coeffs)

        return cluster_coeffs

    @property
    def cluster_ids(self):
        """
        Returns
        -------
        cluster_ids : ndarray
            Cluster cluster_id for each gen_gid
        """
        cluster_ids = None
        if 'cluster_id' in self._meta:
            cluster_ids = self._meta['cluster_id'].values
        return cluster_ids

    @property
    def cluster_coordinates(self):
        """
        Returns
        -------
        cluster_coords : ndarray
            lon, lat coordinates of the centroid of each cluster
        """
        cluster_coords = None
        if 'cluster_id' in self._meta:
            cluster_coords = self._meta.groupby('cluster_id')
            cluster_coords = cluster_coords[['longitude', 'latitude']].mean()
            cluster_coords = cluster_coords.values

        return cluster_coords

    @property
    def coordinates(self):
        """
        Returns
        -------
        coords : ndarray
            lon, lat coordinates for each gen_gid
        """
        coords = self._meta[['longitude', 'latitude']].values
        return coords

    @staticmethod
    def _parse_data(cf_fpath, gen_gids):
        """
        Extract lat, lon coordinates for given gen_gids
        Extract and convert cf_profiles into wavelet coefficients

        Parameters
        ----------
        cf_fpath : str
            Path to reV .h5 files containing desired capacity factor profiles
        gen_gids : list | ndarray
            List or vector of gen_gids to cluster on
        """

        with Outputs(cf_fpath, mode='r', unscale=False) as cfs:
            meta = cfs.meta.loc[gen_gids, ['latitude', 'longitude']]
            gid_slice, gid_idx = RPMClusters._gid_pos(gen_gids)
            coeff = cfs['cf_profile', :, gid_slice][:, gid_idx]

        meta['gen_gid'] = gen_gids
        cols = ['gen_gid', 'latitude', 'longitude']
        meta = meta[cols].reset_index(drop=True)
        coeff = RPMClusters._calculate_wavelets(coeff.T)
        return meta, coeff

    @staticmethod
    def _gid_pos(gen_gids):
        """
        Parameters
        ----------
        gen_gids : list | ndarray
            List or vector of gen_gids to cluster on

        Returns
        -------
        gid_slice : slice
            Slice that encompasses the entire gen_gid range
        gid_idx : ndarray
            Adjusted list to extract gen_gids of interest from slice
        """
        if isinstance(gen_gids, list):
            gen_gids = np.array(gen_gids)

        s = gen_gids.min()
        e = gen_gids.max() + 1
        gid_slice = slice(s, e, None)
        gid_idx = gen_gids - s

        return gid_slice, gid_idx

    @staticmethod
    def _calculate_wavelets(ts_arrays):
        """ Calculates the wavelet coefficients of each
            timeseries within ndarray """
        coefficients = RPMWavelets.get_dwt_coefficients(ts_arrays)
        return coefficients

    def _cluster_coefficients(self, method="kmeans", **kwargs):
        """ Apply a clustering method to <self.ts_arrays> """
        logger.debug('Applying {} clustering '.format(method))

        c_func = getattr(ClusteringMethods, method)
        labels = c_func(self.coefficients, n_clusters=self.n_clusters,
                        **kwargs)
        return labels

    def _dist_rank_optimization(self, norm=None):
        """
        Re-cluster data by minimizing the sum of the:
        - distance between each point and each cluster centroid
        - distance between each point and each

        Parameters
        ----------
        norm : str
            Normalization method to use (see sklearn.preprocessing.normalize)
            if None range normalize

        Returns
        -------
        new_labels : ndarray
            New cluster labels
        """
        cluster_coeffs = self.cluster_coefficients
        cluster_centroids = self.cluster_coordinates
        rmse = []
        dist = []
        for centroid, rep_coeffs in zip(cluster_centroids, cluster_coeffs):
            c_rmse = np.mean((self.coefficients - rep_coeffs) ** 2,
                             axis=1) ** 0.5
            rmse.append(c_rmse)
            c_dist = np.linalg.norm(self.coordinates - centroid, axis=1)
            dist.append(c_dist)

        rmse = ClusteringMethods._normalize_values(np.array(rmse), norm=norm)
        dist = ClusteringMethods._normalize_values(np.array(dist), norm=norm)
        err = (dist**2 + rmse**2)
        new_labels = np.argmin(err, axis=0)
        return new_labels

    def _dist_rank_filter(self, iterate=True, norm=None):
        """
        Re-cluster data by minimizing the sum of the:
        - distance between each point and each cluster centroid
        - distance between each point and each

        Parameters
        ----------
        iterate : bool
            Iterate on _dist_rank_optimization until cluster centroids and
            profiles start to converge
        norm : str
            Normalization method to use (see sklearn.preprocessing.normalize)
            if None range normalize

        Returns
        -------
        new_labels : ndarray
            New cluster labels
        """
        clusters = deepcopy(self)
        coeffs = clusters.cluster_coefficients
        centroids = clusters.cluster_coordinates
        dist, rmse = 0, 0
        while True:
            new_labels = clusters._dist_rank_optimization(norm=norm)
            clusters._meta['cluster_id'] = new_labels
            if iterate:
                c_coeffs = clusters.cluster_coefficients
                c_centroids = clusters.cluster_coordinates
                dist_i = np.linalg.norm(c_centroids - centroids)
                rmse_i = np.mean((c_coeffs - coeffs) ** 2) ** 0.5
                if (dist_i <= dist and rmse_i <= rmse):
                    break
                else:
                    dist, rmse = dist_i, rmse_i
                    coeffs, centroids = c_coeffs, c_centroids
            else:
                break

        return new_labels

    @staticmethod
    def _get_cluster_geom(gdf_points):
        """
        Generate cluster polygons as a geopandas dataframe
        """
        lookup = gdf_points[['latitude', 'longitude']]
        tree = cKDTree(lookup)
        dists, _ = tree.query(lookup, k=2)
        mean_dist = dists.T[1].mean()
        gdf_poly = gdf_points.copy()
        gdf_poly.geometry = gdf_poly.geometry.buffer(mean_dist)
        clusters = gdf_poly.dissolve(by='cluster_id').reset_index()
        clusters.geometry = clusters.geometry.buffer(-mean_dist / 2)

        return clusters, mean_dist

    @staticmethod
    def _generate_shapefile(meta, fpath, beautify=True):
        """
        Generate cluster polygons and save to shapefile
        """
        geometry = [Point(xy) for xy in zip(meta.longitude, meta.latitude)]
        gdf_points = gpd.GeoDataFrame(meta, geometry=geometry,
                                      crs={'init': 'epsg:4326'})

        clusters, mean_dist = RPMClusters._get_cluster_geom(gdf_points)

        if beautify:
            clusters.geometry = clusters.geometry.buffer(-mean_dist)
            clusters.geometry = clusters.geometry.buffer(mean_dist)
            for index, _ in clusters.iterrows():
                geom = clusters.loc[index, 'geometry']
                if geom.geom_type == 'MultiPolygon':
                    clusters.loc[index, 'geometry'] = max(geom,
                                                          key=lambda a: a.area)

        clusters[['cluster_id', 'geometry']].to_file(fpath)
        return fpath

    def _contiguous_filter(self, drop_islands=True, buffer_weight=2):
        """
        Re-classify clusters by making contigous cluster polygons
        """
        meta = self._meta

        geometry = [Point(xy) for xy in zip(meta.longitude, meta.latitude)]
        gdf_points = gpd.GeoDataFrame(meta, geometry=geometry)

        clusters, mean_dist = self._get_cluster_geom(gdf_points)

        # Drop Islands
        if drop_islands:
            buffer = buffer_weight * mean_dist
            clusters.geometry = clusters.geometry.buffer(-buffer)
            clusters.geometry = clusters.geometry.buffer(buffer)
            for index, _ in clusters.iterrows():
                geom = clusters.loc[index, 'geometry']
                if geom.geom_type == 'MultiPolygon':
                    clusters.loc[index, 'geometry'] = max(geom,
                                                          key=lambda a: a.area)

        intersected = gpd.sjoin(gdf_points, clusters,
                                how="left", op='intersects')

        # drop duplicate rows
        gid_counts = intersected.groupby('gen_gid_left').size()
        duplicate_gids = gid_counts[gid_counts > 1].index
        mask = intersected['gen_gid_left'].isin(duplicate_gids)
        intersected.loc[mask, 'cluster_id_right'] = None
        intersected = intersected.drop_duplicates(subset=['gen_gid_left'])

        mask = np.isnan(intersected.cluster_id_right)
        assigned = intersected[~mask].reset_index()
        unassigned = intersected[mask]

        lookup = assigned[['latitude_left', 'longitude_left']]
        target = unassigned[['latitude_left', 'longitude_left']]
        tree = cKDTree(lookup)
        _, inds = tree.query(target, k=1)

        for i, ind in enumerate(list(unassigned.index)):
            nearest_cluster_id = assigned.loc[inds[i], 'cluster_id_left']
            intersected.loc[ind, 'cluster_id_left'] = nearest_cluster_id

        new_labels = intersected['cluster_id_left']

        return new_labels

    def _calculate_ranks(self):
        """ Determine the rank of each location within all clusters
        based on the mean square errors """
        cluster_coeffs = self.cluster_coefficients
        for i, cdf in self.meta.groupby('cluster_id'):
            pos = cdf.index
            rep_coeffs = cluster_coeffs[i]
            coeffs = self.coefficients[pos]
            err = np.mean((coeffs - rep_coeffs) ** 2, axis=1) ** 0.5
            rank = np.argsort(err)
            self._meta.loc[pos, 'rank'] = rank

    def _cluster(self, method='kmeans', method_kwargs=None,
                 dist_rank_filter=True, dist_rmse_kwargs=None,
                 contiguous_filter=True, contiguous_kwargs=None):
        """
        Run three step RPM clustering procedure:
        1) Cluster on wavelet coefficients
        2) Clean up clusters by optimizing rmse and distance
        3) Remove islands using polygon intersection

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

        if self.n_clusters <= 1:
            dist_rank_filter = False
            contiguous_filter = False

        if method_kwargs is None:
            method_kwargs = {}

        labels = self._cluster_coefficients(method=method, **method_kwargs)
        self._meta['cluster_id'] = labels

        # Optimize Distance & Rank
        if dist_rank_filter is True:
            if dist_rmse_kwargs is None:
                dist_rmse_kwargs = {}

            new_labels = self._dist_rank_filter(**dist_rmse_kwargs)
            self._meta['cluster_id'] = new_labels

        # Apply contiguous filter
        if contiguous_filter is True:
            if contiguous_kwargs is None:
                contiguous_kwargs = {}

            new_labels = self._contiguous_filter(**contiguous_kwargs)
            self._meta['cluster_id'] = new_labels

        self._calculate_ranks()

    @classmethod
    def cluster(cls, cf_h5_path, region_gen_gids, n_clusters, method='kmeans',
                method_kwargs=None, dist_rank_filter=True,
                dist_rmse_kwargs=None, contiguous_filter=True,
                contiguous_kwargs=None):
        """
        Entry point for RPMCluster to get clusters for a given region
        defined as a list | array of gen_gids

        Parameters
        ----------
        cf_h5_path : str
            Path to reV .h5 files containing desired capacity factor profiles
        region_gen_gids : list | ndarray
            List or vector of gen_gids to cluster on
        n_clusters : int
            Number of clusters to identify
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

        Returns
        -------
        out : pandas.DataFrame
            Cluster results: (gen_gid, lon, lat, cluster_id, rank)

        Examples
        --------
        >>> from reV import Resource
        >>>
        >>> fname = '$TESTDATADIR/reV_ge/gen_pv_2012.h5'
        >>> with Resource(fname) as res:
        >>>     gen_gids = f.meta.index.values
        >>>
        >>> RPMClusters.cluster(fname, gen_gids, n_clusters=6)
                gen_gid   latitude  longitude  cluster_id   geometry
        0         0  41.290001 -71.860001       0  POINT (-71.86000 41.29000)
        1         1  41.290001 -71.820000       0  POINT (-71.82000 41.29000)
        2         2  41.250000 -71.820000       4  POINT (-71.82000 41.25000)
        3         3  41.330002 -71.820000       0  POINT (-71.82000 41.33000)
        4         4  41.369999 -71.820000       0  POINT (-71.82000 41.37000)
        ..      ...        ...        ...     ...                         ...
        95       95  41.250000 -71.660004       4  POINT (-71.66000 41.25000)
        96       96  41.889999 -71.660004       5  POINT (-71.66000 41.89000)
        97       97  41.450001 -71.660004       3  POINT (-71.66000 41.45000)
        98       98  41.610001 -71.660004       1  POINT (-71.66000 41.61000)
        99       99  41.410000 -71.660004       3  POINT (-71.66000 41.41000)
        """
        clusters = cls(cf_h5_path, region_gen_gids, n_clusters)
        try:
            clusters._cluster(method=method, method_kwargs=method_kwargs,
                              dist_rank_filter=dist_rank_filter,
                              dist_rmse_kwargs=dist_rmse_kwargs,
                              contiguous_filter=contiguous_filter,
                              contiguous_kwargs=contiguous_kwargs)
        except Exception as e:
            logger.exception('Clustering failed on gen_gids {} through {}: {}'
                             .format(np.min(region_gen_gids),
                                     np.max(region_gen_gids), e))
        return clusters.meta


class RPMWavelets:
    """Base class for RPM wavelets"""

    @classmethod
    def get_dwt_coefficients(cls, x, wavelet='Haar', level=None, indices=None):
        """
        Collect wavelet coefficients for time series <x> using
        mother wavelet <wavelet> at levels <level>.

        Parameters
        ----------
        x : ndarray
            time series values
        wavelet : string
            mother wavelet type
        level : int
            optional wavelet computation level
        indices : ndarray
            coefficient array levels to keep

        Returns
        -------
        list
            stacked coefficients at <indices>
        """

        # set mother
        _wavelet = pywt.Wavelet(wavelet)

        # multi-level with default depth
        logger.debug('Calculating wavelet coefficients'
                     ' with {w} wavelet'.format(w=_wavelet.family_name))

        _wavedec = pywt.wavedec(data=x, wavelet=_wavelet, axis=1, level=level)

        return cls._subset_coefficients(x=_wavedec,
                                        gid_count=x.shape[0],
                                        indices=indices)

    @staticmethod
    def _subset_coefficients(x, gid_count, indices=None):
        """
        Subset and stack wavelet coefficients

        Parameters
        ----------
        x : ndarray
            coefficients arrays
        gid_count : int
            number of area ID values
        indices : ndarray
            coefficient array levels to keep

        Returns
        -------
        ndarray
            stacked coefficients: converted to integers
        """

        indices = indices or range(0, len(x))

        _coefficient_count = 0
        for _index in indices:
            _shape = x[_index].shape
            _coefficient_count += _shape[1]

        _combined_wc = np.empty(shape=(gid_count, _coefficient_count),
                                dtype=np.int32)

        logger.debug('{c:d} coefficients'.format(c=_coefficient_count))

        _i_start = 0
        for _index in indices:
            _i_end = _i_start + x[_index].shape[1]
            _combined_wc[:, _i_start:_i_end] = np.round(x[_index])
            _i_start = _i_end

        return _combined_wc
