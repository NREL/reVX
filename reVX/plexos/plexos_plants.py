# -*- coding: utf-8 -*-
"""
Module to create wind and solar plants for PLEXOS nodes
"""
from concurrent.futures import as_completed
import logging
import numpy as np
import os
from sklearn.metrics.pairwise import haversine_distances

from rex.utilities import parse_table, SpawnProcessPool

from reVX.handlers.sc_points import SupplyCurvePoints

logger = logging.getLogger(__name__)


class PlexosPlants:
    """
    Identification and aggregation of renewable resource to Plexos "plants"
    """
    def __init__(self, plexos_table, sc_table, res_meta):
        """
        Parameters
        ----------
        plexos_table : str | pandas.DataFrame
            PLEXOS table of node locations and capacity provided as a .csv,
            .json, or pandas DataFrame
        sc_table : str | pandas.DataFrame
            Supply Curve table .csv or pre-loaded pandas DataFrame
        res_meta : str | pandas.DataFrame
            Path to resource .h5, generation .h5, or pre-extracted .csv or
            pandas DataFrame
        """
        self._plexos_table = self._parse_plexos_table(plexos_table)
        self._sc_points = SupplyCurvePoints(sc_table, res_meta)

        self._capacity = self._plexos_table['capacity'].values
        self._plants = np.full(len(self), None)

        self._sc_node_dist = None

    def __repr__(self):
        msg = "{} with {} plants".format(self.__class__.__name__, len(self))
        return msg

    def __len__(self):
        return len(self._plexos_table)

    def __getitem__(self, plant_id):
        return self._plants[plant_id]

    def __setitem__(self, plant_id, plant_table):
        self._plants[plant_id] = plant_table

    @property
    def plexos_table(self):
        """
        Parsed and reduced PLEXOS table of node locations and capacities

        Returns
        -------
        pandas.DataFrame
        """
        return self._plexos_table

    @property
    def sc_table(self):
        """
        Supply Curve Table

        Returns
        -------
        pandas.DataFrame
        """
        return self._sc_points.sc_table

    @property
    def sc_points(self):
        """
        Supply Curve Points

        Returns
        -------
        SupplyCurvePoints
        """
        return self._sc_points

    @property
    def plant_capacity(self):
        """
        Plant capacities

        Returns
        -------
        ndarray
        """
        return self._capacity

    @property
    def plants(self):
        """
        PLEXOS Plants

        Returns
        -------
        ndarray
        """
        return self._plants

    @property
    def sc_node_dist(self):
        """
        Compute distance between every Supply Curve gid and every PLEXOS node

        Returns
        -------
        ndarray
        """
        if self._sc_node_dist is None:
            cols = ['latitude', 'longitude']
            plant_coords = self._plexos_table[cols].values.astype(float)
            sc_coords = self.sc_table[cols].values.astype(float)

            self._sc_node_dist = PlexosPlants._haversine_dist(plant_coords,
                                                              sc_coords).T

        return self._sc_node_dist

    @staticmethod
    def _parse_plexos_table(plexos_table):
        """
        Parse PLEXOS table from file and reduce table to neccesary columns:
        - latitude
        - longitude
        - capacity

        Parameters
        ----------
        plexos_table : str | pandas.DataFrame
            PLEXOS table of node locations and capacity provided as a .csv,
            .json, or pandas DataFrame

        Returns
        -------
        plexos_table : pandas.DataFrame
            Parsed and reduced PLEXOS table
        """
        plexos_table = parse_table(plexos_table)
        plexos_table = \
            plexos_table.groupby(['latitude', 'longitude'])['capacity'].sum()

        return plexos_table.reset_index()

    @staticmethod
    def _check_coords(coords):
        """
        Check coordinate dimensions and units

        Parameters
        ----------
        coords : ndarray
            Either a single set or an array of (lat, lon) coordinates

        Returns
        -------
        coords : ndarray
            Coordinates in radians
        """
        if len(coords.shape) == 1:
            coords = np.expand_dims(coords, axis=0)

        if np.max(coords) > np.pi or np.min(coords) < - np.pi:
            coords = np.radians(coords)

        return coords

    @staticmethod
    def _haversine_dist(plant_coords, sc_coords):
        """
        Compute the haversine distance between the given plant(s) and given
        supply curve points

        Parameters
        ----------
        plant_coords : ndarray
            (lat, lon) coordinates of plant(s)
        sc_coords : ndarray
            n x 2 array of supply curve (lat, lon) coordinates

        Returns
        -------
        dist : ndarray
            Vector of distances between plant and supply curve points in km
        """
        plant_coords = PlexosPlants._check_coords(plant_coords)
        sc_coords = PlexosPlants._check_coords(sc_coords)

        dist = haversine_distances(plant_coords, sc_coords)
        if plant_coords.shape[0] == 1:
            dist = dist.flatten()

        R = 6373.0  # radius of the earth in kilometers

        return dist * R

    @staticmethod
    def _substation_distance(sc_table, percentile=90):
        """
        Determine the nth percentile of distance between substations and
        transmission from supply curve table

        Parameters
        ----------
        sc_table : pandas.DataFrame
            Supply curve table
        percentile : int, optional
            Percentile to compute substation to transmission distance for,
            by default 90

        Returns
        -------
        dist
            Nth percentile of distance between substations and transmission in
            km, used as plant search distance threshold
        """
        substations = sc_table['trans_type'] == "Substation"
        dist = sc_table.loc[substations, 'dist_mi'].values * 1.6

        return np.percentile(dist, percentile)

    @staticmethod
    def _get_plant_sc_dists(node_coords, sc_table, dist_percentile=90,
                            lcoe_col='total_lcoe', lcoe_thresh=1.3):
        """
        Extract Supply curve gids and distances for plant originating at
        PLEXOS node coords

        Parameters
        ----------
        node_coords : ndarray
            node (lat, lon) coordinates
        sc_table : pandas.DataFrame
            Supply Curve Table
        dist_percentile : int, optional
            Percentile to use to compute distance threshold using sc_gid to
            SubStation distance , by default 90
        lcoe_col : str, optional
            LCOE column to sort by, by default 'total_lcoe'
        lcoe_thresh : float, optional
            LCOE threshold multiplier, exclude sc_gids above threshold,
            by default 1.3

        Returns
        -------
        plant_sc : pandas.DataFrame
            Supply Curve for plant with distance to each sc_gid appended
        """
        sc_coords = np.radians(sc_table[['latitude', 'longitude']].values)

        # Filter SC table to points within 'dist_tresh' of coords
        dist = PlexosPlants._haversine_dist(node_coords, sc_coords)
        dist_thresh = \
            PlexosPlants._substation_distance(sc_table,
                                              percentile=dist_percentile)
        dist_thresh = dist <= dist_thresh
        plant_sc = sc_table[['latitude', 'longitude', lcoe_col]].copy()
        plant_sc = plant_sc.loc[dist_thresh]

        # Find lowest lcoe site
        pos = np.argmin(plant_sc[lcoe_col])
        lcoe_thresh = plant_sc.iloc[pos][lcoe_col] * lcoe_thresh
        plant_coords = \
            plant_sc.iloc[pos][['latitude', 'longitude']].values.astype(float)

        # Filter SC table to lcoe values within 'lcoe_thresh' of min LCOE value
        sc_cols = ['sc_gid', lcoe_col]
        plant_sc = sc_table[sc_cols].copy()
        plant_sc["node_dist"] = dist
        mask = plant_sc[lcoe_col] <= lcoe_thresh
        plant_sc = plant_sc.loc[mask]
        sc_coords = sc_coords[mask]

        # Sort by distance
        plant_sc['dist'] = PlexosPlants._haversine_dist(plant_coords,
                                                        sc_coords)
        plant_sc = plant_sc.sort_values('dist')

        return plant_sc.reset_index(drop=True)

    def _identify_plants(self, dist_percentile=90, lcoe_col='total_lcoe',
                         lcoe_thresh=1.3, max_workers=None):
        """
        Identify plant associated with each node and return supply curve table

        Parameters
        ----------
        dist_percentile : int, optional
            Percentile to use to compute distance threshold using sc_gid to
            SubStation distance , by default 90
        lcoe_col : str, optional
            LCOE column to sort by, by default 'total_lcoe'
        lcoe_thresh : float, optional
            LCOE threshold multiplier, exclude sc_gids above threshold,
            by default 1.3
        max_workers : int, optional
            Number of workers to use for plant sc extraction, 1 == serial,
            > 1 == parallel, None == parallel using all available cpus,
            by default None

        Returns
        -------
        plants : list
            List of plant supply curve tables
        """
        if max_workers is None:
            max_workers = os.cpu_count()

        plants = []
        if max_workers > 1:
            loggers = [__name__, 'reVX']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                futures = []
                for i, node in self.plexos_table.iterrows():
                    coords = \
                        node[['latitude', 'longitude']].values.astype(float)
                    future = exe.submit(PlexosPlants._get_plant_sc_dists,
                                        coords, self.sc_table,
                                        dist_percentile=dist_percentile,
                                        lcoe_col=lcoe_col,
                                        lcoe_thresh=lcoe_thresh)
                    futures.append(future)

                for i, future in enumerate(as_completed(futures)):
                    plants.append(future.result())
                    logger.debug('Completed {} out of {} plant futures.'
                                 .format(i + 1, len(futures)))
        else:
            for i, node in self.plexos_table.iterrows():
                coords = \
                    node[['latitude', 'longitude']].values.astype(float)
                plant = PlexosPlants._get_plant_sc_dists(
                    coords, self.sc_table,
                    dist_percentile=dist_percentile,
                    lcoe_col=lcoe_col,
                    lcoe_thresh=lcoe_thresh)
                plants.append(plant)
                logger.debug('Completed {} out of {} plant futures.'
                             .format(i + 1, len(self)))

        return plants

    @staticmethod
    def _get_sc_gids(plants, idx):
        sc_gids = []
        node_dists = []
        dists = []
        for plant in plants:
            sc_point = plant.iloc[idx]
            sc_gids.append(sc_point['sc_gid'])
            node_dists.append(sc_point['node_dist'])
            dists.append(sc_point['dist'])

        node_dists = np.array(node_dists)
        dists = np.array(dists)

        return sc_gids, dists, node_dists

    def _allocate_sc_gids(self, sc_gids, dists, node_dists):
        unique_gids, plant_gids = np.unique(sc_gids, return_inverse=True)
        for i, sc_gid in enumerate(unique_gids):
            sc_gid = int(sc_gid)
            if self.sc_points.check_sc_gid(sc_gid):
                plant_ids = np.where(plant_gids == i)[0]
                if len(plant) > 1:
                    sc_dists = dists[plant_ids]
                    if len(sc_dists) != len(np.unique(sc_dists)):
                        idxs = np.argsort(node_dists[plant_ids])
                    else:
                        idxs = np.argsort(sc_dists)

                    plant_ids = plant_ids[idxs]

                for plant_id in plant_ids:
                    capacity = self.plant_capacity[plant_id]
                    if self.sc_points.check_sc_gid(sc_gid):
                        sc_point, sc_capacity = \
                            self.sc_points.get_capacity(sc_gid, capacity)

                        plant = self[plant_id]
                        if plant is None:
                            plant = [sc_point]
                        else:
                            plant.append(sc_point)

                        self[plant_id] = plant
                        self._capacity[plant_id] -= sc_capacity
                        logger.info('Allocating {}MW to plant {} from sc_gid '
                                    '{}'.format(sc_capacity, plant_id, sc_gid))

    def _fill_plants(self, plants):
        i = 0
        total_cap = np.sum(self.plant_capacity)
        while np.any(self.plant_capacity > 0):
            plant_cap = np.sum(self.plant_capacity)
            logger.info('Allocating sc_gids to plants round {}'
                        .format(i))
            sc_gids, dists, node_dists = self._get_sc_gids(plants, i)
            self._allocate_sc_gids(sc_gids, dists, node_dists)
            i += 1
            logger.info('{} MW allocated out of {} MW'
                        .format(plant_cap - np.sum(self.plant_capacity),
                                total_cap))
