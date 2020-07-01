# -*- coding: utf-8 -*-
"""
Module to create wind and solar plants for PLEXOS buses
"""
import logging
import numpy as np
import os
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from warnings import warn

from rex.utilities import parse_table, SpawnProcessPool

from reVX.handlers.sc_points import Point, SupplyCurvePoints

logger = logging.getLogger(__name__)


class PlexosPlants:
    """
    Identification and aggregation of renewable resource to Plexos "plants"
    """
    PLEXOS_COLUMNS = ['generator', 'busid', 'busname', 'capacity', 'latitude',
                      'longitude', 'system']

    def __init__(self, plexos_table, sc_table, res_meta, max_workers=None,
                 points_per_worker=400, offshore=False):
        """
        Parameters
        ----------
        plexos_table : str | pandas.DataFrame
            PLEXOS table of bus locations and capacity provided as a .csv,
            .json, or pandas DataFrame
        sc_table : str | pandas.DataFrame
            Supply Curve table .csv or pre-loaded pandas DataFrame
        res_meta : str | pandas.DataFrame
            Path to resource .h5, generation .h5, or pre-extracted .csv or
            pandas DataFrame
        max_workers : int, optional
            Number of workers to use for point and plant creation, 1 == serial,
            > 1 == parallel, None == parallel using all available cpus,
            by default None
        points_per_worker : int, optional
            Number of points to create on each worker, by default 400
        offshore : bool, optional
            Include offshore points, by default False
        """
        if max_workers is None:
            max_workers = os.cpu_count()

        self._max_workers = max_workers
        self._plexos_table = self._parse_plexos_table(plexos_table)
        self._plant_table = None
        self._sc_points = SupplyCurvePoints(
            sc_table, res_meta, max_workers=max_workers,
            points_per_worker=points_per_worker, offshore=offshore)

        self._capacity = self.plant_table['plant_capacity'].values
        self._plants = np.full(len(self), None)

        self._sc_bus_dist = None

    def __repr__(self):
        msg = "{} with {} plants".format(self.__class__.__name__, len(self))
        return msg

    def __len__(self):
        return len(self.plant_table)

    def __getitem__(self, plant_id):
        return self._plants[plant_id]

    def __setitem__(self, plant_id, plant_table):
        self._plants[plant_id] = plant_table

    @property
    def plexos_table(self):
        """
        PLEXOS table

        Returns
        -------
        pandas.DataFrame
        """
        return self._plexos_table

    @property
    def plant_table(self):
        """
        Table of unique PLEXOS plants

        Returns
        -------
        pandas.DataFrame
        """
        if self._plant_table is None:
            self._plant_table = self.plexos_table.drop_duplicates(
                'plant_id').sort_values('plant_id')

            cols = ['latitude', 'longitude', 'plant_id', 'plant_capacity']
            self._plant_table = self._plant_table[cols].set_index('plant_id')

        return self._plant_table

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
    def sc_bus_dist(self):
        """
        Compute distance between every Supply Curve gid and every PLEXOS bus

        Returns
        -------
        ndarray
        """
        if self._sc_bus_dist is None:
            cols = ['latitude', 'longitude']
            plant_coords = self._plexos_table[cols].values.astype(float)
            sc_coords = self.sc_table[cols].values.astype(float)

            self._sc_bus_dist = PlexosPlants._haversine_dist(plant_coords,
                                                             sc_coords).T

        return self._sc_bus_dist

    def _parse_plexos_table(self, plexos_table):
        """
        Parse PLEXOS table from file and reduce to PLEXOS_COLS
        Combine buses at the same coordinates and add unique plant_ids

        Parameters
        ----------
        plexos_table : str | pandas.DataFrame
            PLEXOS table of bus locations and capacity provided as a .csv,
            .json, or pandas DataFrame

        Returns
        -------
        plexos_table : pandas.DataFrame
            Parsed and reduced PLEXOS table
        """
        plexos_table = parse_table(plexos_table)
        cols = [c for c in plexos_table if c.lower() in self.PLEXOS_COLUMNS]
        plexos_table = plexos_table[cols]
        rename = {}
        for c in plexos_table:
            if c.lower() == 'capacity':
                rename[c] = 'capacity'
            elif c.lower() == 'latitude':
                rename[c] = 'latitude'
            elif c.lower() == 'longitude':
                rename[c] = 'longitude'

        plexos_table = plexos_table.rename(columns=rename)

        mask = plexos_table['latitude'] > 90
        mask |= plexos_table['latitude'] < -90
        mask |= plexos_table['longitude'] > 180
        mask |= plexos_table['longitude'] < -180
        if np.any(mask):
            msg = ('WARNING: {} Buses have invalid coordinates:\n{}'
                   .format(np.sum(mask), plexos_table.loc[mask]))
            logger.warning(msg)
            warn(msg)
            plexos_table = plexos_table.loc[~mask]

        mask = plexos_table['capacity'] > 0
        cols = ['latitude', 'longitude']
        plant_cap = \
            plexos_table.loc[mask].groupby(cols)['capacity'].sum()
        plant_cap = plant_cap.reset_index().reset_index()
        rename = {'index': 'plant_id', 'capacity': 'plant_capacity'}
        plant_cap = plant_cap.rename(columns=rename)

        plexos_table = plexos_table.merge(plant_cap,
                                          on=cols,
                                          how='outer')

        return plexos_table

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
    def _get_plant_sc_dists(bus_coords, sc_table, dist_percentile=90,
                            lcoe_col='total_lcoe', lcoe_thresh=1.3):
        """
        Extract Supply curve gids and distances for plant originating at
        PLEXOS bus coords

        Parameters
        ----------
        bus_coords : ndarray
            bus (lat, lon) coordinates
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
        logger.debug("Extracting supply curve gids for bus at {}"
                     .format(bus_coords))
        sc_coords = np.radians(sc_table[['latitude', 'longitude']].values)

        # Filter SC table to points within 'dist_tresh' of coords
        dist = PlexosPlants._haversine_dist(bus_coords, sc_coords)
        dist_thresh = \
            PlexosPlants._substation_distance(sc_table,
                                              percentile=dist_percentile)
        logger.debug("- Using distance threshold of {} km".format(dist_thresh))
        while True:
            mask = dist <= dist_thresh
            plant_sc = sc_table[['latitude', 'longitude', lcoe_col]].copy()
            plant_sc = plant_sc.loc[mask]
            if len(plant_sc) > 1:
                break
            else:
                dist_thresh *= 1.2

        # Find lowest lcoe site
        pos = np.argmin(plant_sc[lcoe_col])
        lcoe_thresh = plant_sc.iloc[pos][lcoe_col] * lcoe_thresh
        plant_coords = \
            plant_sc.iloc[pos][['latitude', 'longitude']].values.astype(float)
        logger.debug("- Plant will be centered at {}".format(plant_coords))
        logger.debug("- Only supply curve points with an lcoe < {} will be "
                     "used".format(lcoe_thresh))

        # Filter SC table to lcoe values within 'lcoe_thresh' of min LCOE value
        sc_cols = ['sc_gid', lcoe_col]
        plant_sc = sc_table[sc_cols].copy()
        plant_sc["bus_dist"] = dist
        mask = plant_sc[lcoe_col] <= lcoe_thresh
        plant_sc = plant_sc.loc[mask]
        sc_coords = sc_coords[mask]

        # Sort by distance
        plant_sc['dist'] = PlexosPlants._haversine_dist(plant_coords,
                                                        sc_coords)
        plant_sc = plant_sc.sort_values('dist')

        return plant_sc.reset_index(drop=True)

    @staticmethod
    def _identify_plants(plant_table, sc_table, dist_percentile=90,
                         lcoe_col='total_lcoe', lcoe_thresh=1.3,
                         max_workers=None, plants_per_worker=40):
        """
        Identify plant associated with each bus and return supply curve table

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
            Number of workers to use for plant identification, 1 == serial,
            > 1 == parallel, None == parallel using all available cpus,
            by default None
        plants_per_worker : int, optional
            Number of plants to identify on each worker, by default 40

        Returns
        -------
        plants : list
            List of plant supply curve tables
        """
        plants = []
        if max_workers is None:
            max_workers = os.cpu_count()

        if max_workers > 1:
            logger.info('Identifying plants in parallel')
            loggers = [__name__, 'reVX']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                futures = []
                slices = Point._create_worker_slices(
                    plant_table, points_per_worker=plants_per_worker)
                for table_slice in slices:
                    future = exe.submit(PlexosPlants._identify_plants,
                                        plant_table.iloc[table_slice],
                                        sc_table,
                                        dist_percentile=dist_percentile,
                                        lcoe_col=lcoe_col,
                                        lcoe_thresh=lcoe_thresh,
                                        max_workers=1)
                    futures.append(future)

                for i, future in enumerate(futures):
                    plants.extend(future.result())
                    logger.debug('Identified {} out of {} plants'
                                 .format((i + 1) * plants_per_worker,
                                         len(plant_table)))
        else:
            logger.info('Identifying plants in serial')
            for i, bus in plant_table.iterrows():
                coords = \
                    bus[['latitude', 'longitude']].values.astype(float)
                plant = PlexosPlants._get_plant_sc_dists(
                    coords, sc_table,
                    dist_percentile=dist_percentile,
                    lcoe_col=lcoe_col,
                    lcoe_thresh=lcoe_thresh)
                plants.append(plant)
                logger.debug('Identified {} out of {} plants.'
                             .format(i + 1, len(plant_table)))

        return plants

    @staticmethod
    def _get_sc_gids(plants, idx):
        """
        For all plants extract sc_gid, dist (to sc_gid), and bus_dist from
        Supply Curve points

        Parameters
        ----------
        plants : list
            List of sc_table subsets for all plants
        idx : int
            index to extract from plant sc_tables

        Returns
        -------
        tuple
            (sc_gids, dists, bus_dists)
        """
        sc_gids = []
        bus_dists = []
        dists = []
        for plant in plants:
            sc_point = plant.iloc[idx]
            sc_gids.append(sc_point['sc_gid'])
            bus_dists.append(sc_point['bus_dist'])
            dists.append(sc_point['dist'])

        bus_dists = np.array(bus_dists)
        dists = np.array(dists)

        return sc_gids, dists, bus_dists

    def _allocate_sc_gids(self, sc_gids, dists, bus_dists):
        """
        Allocate capacity from supply curve points to plants

        Parameters
        ----------
        sc_gids : list
            List of supply curve point gids to allocate capacity from
        dists : list
            List of distances from plants to sc_gids
        bus_dists : list
            List of distances from bus associated with plants to sc_gids
        """
        unique_gids, plant_gids = np.unique(sc_gids, return_inverse=True)
        for i, sc_gid in enumerate(unique_gids):
            sc_gid = int(sc_gid)
            if self.sc_points.check_sc_gid(sc_gid):
                plant_ids = np.where(plant_gids == i)[0]
                if len(plant_ids) > 1:
                    sc_dists = dists[plant_ids]
                    if len(sc_dists) != len(np.unique(sc_dists)):
                        idxs = np.argsort(bus_dists[plant_ids])
                    else:
                        idxs = np.argsort(sc_dists)

                    plant_ids = plant_ids[idxs]

                for plant_id in plant_ids:
                    capacity = self.plant_capacity[plant_id]
                    if (capacity > 0) and self.sc_points.check_sc_gid(sc_gid):
                        sc_point, sc_capacity = \
                            self.sc_points.get_capacity(sc_gid, capacity)
                        if sc_capacity:
                            plant = self[plant_id]
                            if plant is None:
                                plant = [sc_point]
                            else:
                                plant.append(sc_point)

                            self[plant_id] = plant
                            self._capacity[plant_id] -= sc_capacity
                            logger.debug('Allocating {:.1f}MW to plant {} from'
                                         ' sc_gid {}'.format(sc_capacity,
                                                             plant_id,
                                                             sc_gid))
                        else:
                            msg = ('WARNING: sc_gid {} returned 0 capacity!'
                                   .format(sc_gid))
                            logger.warning(msg)
                            warn(msg)

    def _fill_plants(self, plants):
        """
        Fill plants with capacity from supply curve points

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
        i = 0
        total_cap = np.sum(self.plant_capacity)
        while np.any(self.plant_capacity > 0):
            i_cap = np.sum(self.plant_capacity[self.plant_capacity > 0])
            logger.info('Allocating sc_gids to plants round {}'
                        .format(i))
            sc_gids, dists, bus_dists = self._get_sc_gids(plants, i)
            self._allocate_sc_gids(sc_gids, dists, bus_dists)
            cap = np.sum(self.plant_capacity[self.plant_capacity > 0])
            logger.info('{:.1f} MW allocated in round {}'
                        .format(i_cap - cap, i))
            i += 1
            logger.info('{:.1f} MW allocated out of {:.1f} MW'
                        .format(total_cap - cap, total_cap))
            logger.info('{} of {} plants have been filled'
                        .format(np.sum(self.plant_capacity <= 0), len(self)))

    def fill_plants(self, dist_percentile=90, lcoe_col='total_lcoe',
                    lcoe_thresh=1.3, plants_per_worker=40):
        """
        Fill plants with capacity from supply curve points

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
        plants_per_worker : int, optional
            Number of plants to identify on each worker, by default 40

        Returns
        -------
        plants : list
            List of plants being built
        """
        plants = self._identify_plants(self.plant_table, self.sc_table,
                                       dist_percentile=dist_percentile,
                                       lcoe_col=lcoe_col,
                                       lcoe_thresh=lcoe_thresh,
                                       max_workers=self._max_workers,
                                       plants_per_worker=plants_per_worker)
        self._fill_plants(plants)

        plants = [pd.concat(plant, axis=1).T for plant in self.plants]

        return plants

    def plants_meta(self, plants, out_fpath=None):
        """
        Create plants meta data from filled plants DataFrames:
            - Location (lat, lon)
            - final capacity
            - sc_gids
            - res_gids
            - res gid_counts

        Parameters
        ----------
        plants : list
            List of filled plant DataFrames
        out_fpath : str
            .csv path to save plant meta data too

        Returns
        -------
        plants_meta : pandas.DataFrame
            Location (lat, lon), final capacity, and associated sc_gids,
            res_gids, and res gid_counts for all plants
        """
        plants_meta = []
        for i, plant in enumerate(plants):
            plants_meta.append(pd.Series(
                {'sc_gids': plant['sc_gid'].values.tolist(),
                 'res_gids': plant['res_gids'].values.tolist(),
                 'gid_counts': plant['gid_counts'].values.tolist(),
                 'res_cf_means': plant['cf_means'].values.tolist(),
                 'build_capacity': plant['build_capacity'].values.tolist(),
                 'cf_mean': np.hstack(plant['cf_means'].values).mean()},
                name=i))

        plants_meta = pd.concat(plants_meta, axis=1).T
        plants_meta.index.name = 'plant_id'

        plants_meta = self.plexos_table.merge(plants_meta.reset_index(),
                                              on='plant_id', how='outer')

        if out_fpath:
            plants_meta.to_csv(out_fpath, index=False)

        return plants_meta

    @classmethod
    def fill(cls, plexos_table, sc_table, res_meta, dist_percentile=90,
             lcoe_col='total_lcoe', lcoe_thresh=1.3, max_workers=None,
             points_per_worker=400, plants_per_worker=40, offshore=False,
             out_fpath=None):
        """
        Fill plants with capacity from supply curve points

        Parameters
        ----------
        plexos_table : str | pandas.DataFrame
            PLEXOS table of bus locations and capacity provided as a .csv,
            .json, or pandas DataFrame
        sc_table : str | pandas.DataFrame
            Supply Curve table .csv or pre-loaded pandas DataFrame
        res_meta : str | pandas.DataFrame
            Path to resource .h5, generation .h5, or pre-extracted .csv or
            pandas DataFrame
        dist_percentile : int, optional
            Percentile to use to compute distance threshold using sc_gid to
            SubStation distance , by default 90
        lcoe_col : str, optional
            LCOE column to sort by, by default 'total_lcoe'
        lcoe_thresh : float, optional
            LCOE threshold multiplier, exclude sc_gids above threshold,
            by default 1.3
        max_workers : int, optional
            Number of workers to use for point and plant creation, 1 == serial,
            > 1 == parallel, None == parallel using all available cpus,
            by default None
        points_per_worker : int, optional
            Number of points to create on each worker, by default 400
        plants_per_worker : int, optional
            Number of plants to identify on each worker, by default 40
        offshore : bool, optional
            Include offshore points, by default False
        out_fpath : str
            .csv path to save plant meta data too

        Returns
        -------
        plants : list
            List of plant supply curve tables
        plants_meta : pandas.DataFrame
            Plants meta data
        """
        pp = cls(plexos_table, sc_table, res_meta, max_workers=max_workers,
                 points_per_worker=points_per_worker, offshore=offshore)
        plants = pp.fill_plants(dist_percentile=dist_percentile,
                                lcoe_col=lcoe_col,
                                lcoe_thresh=lcoe_thresh,
                                plants_per_worker=plants_per_worker)

        plants_meta = pp.plants_meta(plants, out_fpath=out_fpath)

        return plants, plants_meta
