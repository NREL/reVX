# -*- coding: utf-8 -*-
"""
Module to create wind and solar plants for PLEXOS buses
"""
import json
import logging
import numpy as np
import os
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from warnings import warn

from rex.resource import Resource
from rex.utilities import parse_table, SpawnProcessPool
from rex.utilities.utilities import to_records_array

from reVX.handlers.outputs import Outputs
from reVX.handlers.sc_points import SupplyCurvePoints
from reVX.utilities.utilities import log_versions
from reVX.plexos.base import BaseProfileAggregation, TZ_ALIASES

logger = logging.getLogger(__name__)


class Plants:
    """
    Base class to handle plants
    """

    def __init__(self, plants):
        self._plants = {}

        if isinstance(plants, (np.ndarray, list, tuple)):
            for i, plant in enumerate(plants):
                self[i] = plant
        elif isinstance(plants, dict):
            self._plants = plants

        self._i = 0

    def __repr__(self):
        msg = "{} with {} plants".format(self.__class__.__name__, len(self))
        return msg

    def __len__(self):
        return len(self._plants)

    def __getitem__(self, plant_id):
        """
        Get the plant build out for the given plant_id

        Parameters
        ----------
        plant_id : int | str
            Unique Id for plant of interest

        Returns
        -------
        list
            List of sc_gids' and thier associated resource gids to build plant
            from
        """
        return self._plants.get(plant_id, [])

    def __setitem__(self, plant_id, plant_build):
        """
        Update plant build

        Parameters
        ----------
        plant_id : int | str
            Unique Id for plant to update
        plant_build : list
            List of sc_gids' and thier associated resource gids to build plant
            from
        """
        self._plants[plant_id] = plant_build

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self._plants):
            self._i = 0
            raise StopIteration

        plant_id = self.plant_ids[self._i]
        self._i += 1

        return self[plant_id]

    @property
    def plant_ids(self):
        """
        Plant ids

        Returns
        -------
        list
        """
        return list(self._plants.keys())

    @property
    def plants(self):
        """
        Dictionary matching plants to plant ids

        Returns
        -------
        dict
        """
        return self._plants

    @property
    def plant_builds(self):
        """
        List of plant builds

        Returns
        -------
        dict
        """
        plant_builds = {pid: pd.concat(plant, axis=1).T
                        for pid, plant in self.plants.items()
                        if plant is not None}

        return plant_builds

    @staticmethod
    def _parse_lists(column):
        """
        Check to see if list values are strings, if so parse with json.loads

        Parameters
        ----------
        column : pandas.Series
            Pandas DataFrame column to check

        Returns
        -------
        column : pandas.Series
            Pandas DataFrame column with values converted to lists if needed
        """
        if isinstance(column.iloc[0], str):
            column = column.apply(json.loads).values

        return column

    @classmethod
    def load(cls, plants_fpath):
        """
        Load pre-filled plants from disc

        Parameters
        ----------
        plants_fpath : str | DataFrame
            DataFrame or path to .csv containing pre-filled plants

        Returns
        -------
        Plants
            Initialized Plants instance with pre-filled plants
        """
        plant_builds = parse_table(plants_fpath)
        if 'plant_id' in plant_builds:
            plant_builds = plant_builds.set_index('plant_id')

        plant_builds = plant_builds.apply(cls._parse_lists)

        plants = {}
        for pid, build in plant_builds.iterrows():
            plant = []
            for i in range(len(build['sc_gids'])):
                sc_point = \
                    pd.Series({'sc_gid': build['sc_gids'][i],
                               'res_gids': build['res_gids'][i],
                               'gid_counts': build['gid_counts'][i],
                               'cf_means': build['res_cf_means'][i],
                               'build_capacity': build['build_capacity'][i]})
                plant.append(sc_point)

            plants[pid] = plant

        return cls(plants)


class PlexosPlants(Plants):
    """
    Class to identify and fill Plants
    """

    def __init__(self, plexos_table, sc_table, mymean_fpath,
                 dist_percentile=90, dist_thresh_km=None,
                 lcoe_col='total_lcoe', lcoe_thresh=1.3, offshore=False,
                 max_workers=None, plants_per_worker=40,
                 points_per_worker=400):
        """
        Parameters
        ----------
        plexos_table : pandas.DataFrame
            Parsed and clean PLEXOS table
        sc_table : str | pandas.DataFrame
            Supply Curve table .csv or pre-loaded pandas DataFrame
        mymean_fpath : str
            Path to reV multi-year-mean .h5 file (preferred) or annual reV
            generation output .h5 file. If annual, the plant buildouts might
            change from year to year (bad!).
        offshore : bool, optional
            Include offshore points, by default False
        dist_percentile : int, optional
            Percentile to use to compute distance threshold using sc_gid to
            SubStation distance , by default 90
        dist_thresh_km : float | None
            Optional absolute distance threshold in km that will override the
            dist_percentile input.
        lcoe_col : str, optional
            LCOE column to sort by, by default 'total_lcoe'
        lcoe_thresh : float, optional
            LCOE threshold multiplier, exclude sc_gids above threshold,
            by default 1.3
        offshore : bool, optional
            Include offshore points, by default False
        max_workers : int, optional
            Number of workers to use for plant identification, 1 == serial,
            > 1 == parallel, None == parallel using all available cpus,
            by default None
        plants_per_worker : int, optional
            Number of plants to identify on each worker, by default 40
        points_per_worker : int, optional
            Number of points to create on each worker, by default 400
        """
        log_versions(logger)
        logger.info('Initializing PlexosPlants from plexos table with '
                    '{} rows'.format(len(plexos_table)))
        self._plant_table = self._parse_plant_table(plexos_table)
        self._capacity = self.plant_table['plant_capacity'].values.copy()
        self._cap_to_alloc = self.plant_table['plant_capacity'].values.copy()

        if max_workers is None:
            max_workers = os.cpu_count()

        self._sc_points = \
            SupplyCurvePoints(sc_table, mymean_fpath,
                              max_workers=max_workers,
                              points_per_worker=points_per_worker,
                              offshore=offshore)

        plants = self._identify_plants(self.plant_table,
                                       self._sc_points.sc_table,
                                       dist_percentile=dist_percentile,
                                       dist_thresh_km=dist_thresh_km,
                                       lcoe_col=lcoe_col,
                                       lcoe_thresh=lcoe_thresh,
                                       max_workers=max_workers,
                                       plants_per_worker=plants_per_worker)

        self._plants = {}
        self._fill_plants(plants)

        self._i = 0

    @property
    def plant_table(self):
        """
        Plants meta data table

        Returns
        -------
        pandas.DataFrame
        """
        return self._plant_table

    @property
    def plant_capacity(self):
        """
        Plant capacities in MW

        Returns
        -------
        ndarray
        """
        return self._capacity

    @property
    def sc_points(self):
        """
        Supply Curve Points

        Returns
        -------
        SupplyCurvePoints
        """
        return self._sc_points

    @staticmethod
    def _parse_plant_table(plexos_table):
        """
        Create Table of unique PLEXOS plants from plexos table

        Parameters
        ----------
        plexos_table : pandas.DataFrame
            Parsed and clean PLEXOS table

        Returns
        -------
        plant_table : pandas.DataFrame
            Table of unique plants from plexos table
        """
        plexos_table = parse_table(plexos_table)
        if 'plant_id' not in plexos_table:
            plexos_table = \
                PlantProfileAggregation._parse_plexos_table(plexos_table)

        plant_table = \
            plexos_table.drop_duplicates('plant_id').sort_values('plant_id')

        cols = ['plant_id', 'plant_capacity', 'generator', 'busid', 'busname',
                'capacity', 'latitude', 'longitude', 'system']
        cols = [c for c in cols if c in plant_table]
        plant_table = plant_table[cols].set_index('plant_id')

        return plant_table

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

    @classmethod
    def _haversine_dist(cls, plant_coords, sc_coords):
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
        plant_coords = cls._check_coords(plant_coords)
        sc_coords = cls._check_coords(sc_coords)

        dist = haversine_distances(plant_coords, sc_coords)
        if plant_coords.shape[0] == 1:
            dist = dist.flatten()

        R = 6371.0  # radius of the earth in kilometers

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

        if 'dist_km' in sc_table:
            dist = sc_table.loc[substations, 'dist_km'].values
        elif 'dist_mi' in sc_table:
            dist = sc_table.loc[substations, 'dist_mi'].values * 1.6
        else:
            msg = ('Could not find "dist_mi" or "dist_km" in sc table '
                   'columns: {}'.format(sc_table.columns.values))
            logger.error(msg)
            raise KeyError(msg)

        return np.percentile(dist, percentile)

    @classmethod
    def _get_plant_sc_dists(cls, bus_coords, sc_table,
                            dist_percentile=90, dist_thresh_km=None,
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
        dist_thresh_km : float | None
            Optional absolute distance threshold in km that will override the
            dist_percentile input.
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
        dist_km = cls._haversine_dist(bus_coords, sc_coords)

        if dist_thresh_km is None:
            dist_thresh_km = cls._substation_distance(
                sc_table, percentile=dist_percentile)

        logger.debug("- Using distance threshold of {} km"
                     .format(dist_thresh_km))
        while True:
            mask = dist_km <= dist_thresh_km
            plant_sc = sc_table[['latitude', 'longitude', lcoe_col]].copy()
            plant_sc = plant_sc.loc[mask]
            if len(plant_sc) > 1:
                break
            else:
                dist_thresh_km *= 1.2

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
        plant_sc["bus_dist"] = dist_km
        mask = plant_sc[lcoe_col] <= lcoe_thresh
        plant_sc = plant_sc.loc[mask]
        sc_coords = sc_coords[mask]

        # Sort by distance
        plant_sc['dist'] = cls._haversine_dist(plant_coords, sc_coords)
        plant_sc = plant_sc.sort_values('dist')

        return plant_sc.reset_index(drop=True)

    @classmethod
    def _identify_plants(cls, plant_table, sc_table,
                         dist_percentile=90, dist_thresh_km=None,
                         lcoe_col='total_lcoe', lcoe_thresh=1.3,
                         max_workers=None, plants_per_worker=40):
        """
        Identify plant associated with each bus and return supply curve table

        Parameters
        ----------
        plant_table : pandas.DataFrame
            Table of unique plants from plexos table
        sc_table : str | pandas.DataFrame
            Supply Curve table .csv or pre-loaded pandas DataFrame
        dist_percentile : int, optional
            Percentile to use to compute distance threshold using sc_gid to
            SubStation distance , by default 90
        dist_thresh_km : float | None
            Optional absolute distance threshold in km that will override the
            dist_percentile input.
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
            List of supply curve points that can be used to fill each plant of
            interest
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
                slices = SupplyCurvePoints._create_worker_slices(
                    plant_table, points_per_worker=plants_per_worker)
                for table_slice in slices:
                    future = exe.submit(cls._identify_plants,
                                        plant_table.iloc[table_slice].copy(),
                                        sc_table,
                                        dist_percentile=dist_percentile,
                                        dist_thresh_km=dist_thresh_km,
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
                plant = cls._get_plant_sc_dists(
                    coords, sc_table,
                    dist_percentile=dist_percentile,
                    dist_thresh_km=dist_thresh_km,
                    lcoe_col=lcoe_col,
                    lcoe_thresh=lcoe_thresh)
                plants.append(plant)
                logger.debug('Identified {} out of {} PlexosPlants.'
                             .format(i + 1, len(plant_table)))

        return plants

    @staticmethod
    def _get_sc_gids(identified_plants, idx):
        """
        For all plants extract sc_gid, dist (to sc_gid), and bus_dist from
        Supply Curve points

        Parameters
        ----------
        identified_plants : list
            List of identified plants, I.E., Supply curve points available
            to fill each plant along with the distance to the plant center
            and associated bus.
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
        for plant in identified_plants:
            sc_point = plant.iloc[idx]
            sc_gids.append(sc_point['sc_gid'])
            bus_dists.append(sc_point['bus_dist'])
            dists.append(sc_point['dist'])

        bus_dists = np.array(bus_dists)
        dists = np.array(dists)

        return sc_gids, dists, bus_dists

    def _allocate_sc_gids(self, sc_gids, dists, bus_dists):
        """
        Allocate capacity (MW) from supply curve points to plants

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
                    capacity = self._cap_to_alloc[plant_id]
                    if (capacity > 0) and self.sc_points.check_sc_gid(sc_gid):
                        built_point = self.sc_points.get_capacity(sc_gid,
                                                                  capacity)
                        if built_point is not None:
                            built_cap = built_point['build_capacity']
                            plant = self[plant_id]
                            plant.append(built_point)

                            self[plant_id] = plant
                            self._cap_to_alloc[plant_id] -= built_cap
                            logger.debug('Allocating {:.1f}MW to plant {} from'
                                         ' sc_gid {}'.format(built_cap,
                                                             plant_id,
                                                             sc_gid))
                        else:
                            msg = ('WARNING: sc_gid {} returned 0 capacity!'
                                   .format(sc_gid))
                            logger.warning(msg)
                            warn(msg)

    def _fill_plants(self, identified_plants):
        """
        Fill plants with capacity (MW) from supply curve points

        Parameters
        ----------
        identified_plants : list
            List of identified plants, I.E., Supply curve points available
            to fill each plant along with the distance to the plant center
            and associated bus.
        """
        i = 0
        total_cap = np.sum(self.plant_capacity)
        while np.any(self._cap_to_alloc > 0):
            cap_remaining_0 = sum(self._cap_to_alloc)
            logger.info('Allocating sc_gids to plants round {}'
                        .format(i))
            sc_gids, dists, bus_dists = self._get_sc_gids(identified_plants, i)
            self._allocate_sc_gids(sc_gids, dists, bus_dists)
            cap_remaining_1 = sum(self._cap_to_alloc)
            i_cap_allocated = cap_remaining_0 - cap_remaining_1
            logger.info('{:.1f} MW allocated in round {}'
                        .format(i_cap_allocated, i))
            i += 1
            logger.info('{:.1f} MW allocated out of {:.1f} MW'
                        .format(total_cap - cap_remaining_1, total_cap))
            logger.info('{} of {} plants have been filled'
                        .format(np.sum(self._cap_to_alloc <= 0), len(self)))

    def dump(self, out_fpath=None):
        """
        Create plants meta data from filled plants DataFrames:
            - Location (lat, lon)
            - final capacity (MW)
            - sc_gids
            - res_gids
            - res gid_counts

        Parameters
        ----------
        out_fpath : str, optional
            .csv path to save plant meta data too, by default None

        Returns
        -------
        plants_meta : pandas.DataFrame
            Location (lat, lon), final capacity (MW), and associated sc_gids,
            res_gids, and res gid_counts for all plants
        """
        plants_meta = []
        for pid, plant in self.plant_builds.items():
            plants_meta.append(pd.Series(
                {'sc_gids': plant['sc_gid'].values.tolist(),
                 'res_gids': plant['res_gids'].values.tolist(),
                 'gid_counts': plant['gid_counts'].values.tolist(),
                 'res_cf_means': plant['cf_means'].values.tolist(),
                 'build_capacity': plant['build_capacity'].values.tolist()},
                name=pid))

        plants_meta = pd.concat(plants_meta, axis=1).T
        plants_meta.index.name = 'plant_id'

        if out_fpath:
            plants_meta.to_csv(out_fpath)

        return plants_meta

    @classmethod
    def save(cls, plexos_table, sc_table, mymean_fpath, out_fpath,
             dist_percentile=90, lcoe_col='total_lcoe', lcoe_thresh=1.3,
             offshore=False, max_workers=None, plants_per_worker=40,
             points_per_worker=400):
        """
        Identify, fill, and then save plants to disc

        Parameters
        ----------
        plexos_table : pandas.DataFrame
            Parsed and clean PLEXOS table
        sc_table : str | pandas.DataFrame
            Supply Curve table .csv or pre-loaded pandas DataFrame
        mymean_fpath : str
            Path to reV multi-year-mean .h5 file (preferred) or annual reV
            generation output .h5 file. If annual, the plant buildouts might
            change from year to year (bad!).
        out_fpath : str
            .csv path to save plant meta data too
        offshore : bool, optional
            Include offshore points, by default False
        dist_percentile : int, optional
            Percentile to use to compute distance threshold using sc_gid to
            SubStation distance , by default 90
        lcoe_col : str, optional
            LCOE column to sort by, by default 'total_lcoe'
        lcoe_thresh : float, optional
            LCOE threshold multiplier, exclude sc_gids above threshold,
            by default 1.3
        offshore : bool, optional
            Include offshore points, by default False
        max_workers : int, optional
            Number of workers to use for plant identification, 1 == serial,
            > 1 == parallel, None == parallel using all available cpus,
            by default None
        plants_per_worker : int, optional
            Number of plants to identify on each worker, by default 40
        points_per_worker : int, optional
            Number of points to create on each worker, by default 400
        """
        pp = cls(plexos_table, sc_table, mymean_fpath,
                 dist_percentile=dist_percentile, lcoe_col=lcoe_col,
                 lcoe_thresh=lcoe_thresh, offshore=offshore,
                 max_workers=max_workers, plants_per_worker=plants_per_worker,
                 points_per_worker=points_per_worker)

        pp.dump(out_fpath=out_fpath)


class PlantProfileAggregation:
    """
    Aggregate renewable generation profiles to Plexos "plants"
    """

    def __init__(self, plexos_table, sc_table, mymean_fpath, cf_fpath,
                 plants=None, dist_percentile=90, dist_thresh_km=None,
                 lcoe_col='total_lcoe', lcoe_thresh=1.3,
                 offshore=False, max_workers=None,
                 plants_per_worker=40, points_per_worker=400,
                 plant_name_col=None, tech_tag=None, dset_tag='',
                 timezone='UTC', bespoke=False):
        """
        Parameters
        ----------
        plexos_table : str | pandas.DataFrame
            PLEXOS table of bus locations and capacity (MW) provided as a .csv,
            .json, or pandas DataFrame. Needs columns: latitude, longitude,
            capacity. Note that capacity needs to be AC for wind and DC for
            solar.
        sc_table : str | pandas.DataFrame
            Supply Curve table .csv or pre-loaded pandas DataFrame
        mymean_fpath : str
            Path to reV multi-year-mean output .h5 file to pull cf_mean-means
            from - this will be used to determine where plants are built so it
            does not differ from year to year. The meta must match cf_fpath.
        cf_fpath : str
            Path to reV annual Generation output .h5 file to pull CF profiles
            from. Meta must match mymean_fpath.
        plants : PlexosPlants | None
            Optional PlexosPlants input. If None, PlexosPlants object will
            be Initialized from the plexos table input.
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
        plant_name_col : str | None
            Column in plexos_table that has the plant name that should be used
            in the plexos output csv column headers.
        tech_tag : str | None
            Optional technology tag to include as a suffix in the plexos output
            csv column headers.
        dset_tag : str
            Dataset tag to append to dataset names in cf profile file. e.g. If
            the cf profile file is a multi year file using dset_tag="-2008"
            will enable us to select the corresponding datasets
            (cf_mean-2008, cf_profile-2008, etc)
        timezone : str
            Timezone for output generation profiles. This is a string that will
            be passed to pytz.timezone() e.g. US/Pacific, US/Mountain,
            US/Central, US/Eastern, or UTC. For a list of all available
            timezones, see pytz.all_timezones
        bespoke : bool
            Flag to signify if the cf_fpath file was generated using the reV
            bespoke wind module. The bespoke output files have generation
            profiles at the supply curve grid resolution which is different
            than traditional reV generation outputs that are on the resource
            grid resolution.
        """
        log_versions(logger)
        logger.info('Initializing PlantProfileAggregation')

        self._plexos_table = self._parse_plexos_table(
            plexos_table, plant_name_col=plant_name_col)

        self._plant_name_col = plant_name_col
        self._tech_tag = tech_tag
        self._dset_tag = dset_tag
        self._timezone = timezone
        self._mymean_fpath = mymean_fpath
        self._cf_fpath = cf_fpath
        self._cf_gid_map = self._parse_cf_gid_map(cf_fpath)
        self._sc_table = SupplyCurvePoints._parse_sc_table(sc_table,
                                                           offshore=offshore)

        if bespoke:
            self._sc_table = BaseProfileAggregation.convert_bespoke_sc(
                self._sc_table, 'gid')

        if plants is None:
            self._plants = PlexosPlants(self._plexos_table, self._sc_table,
                                        mymean_fpath,
                                        dist_percentile=dist_percentile,
                                        dist_thresh_km=dist_thresh_km,
                                        lcoe_col=lcoe_col,
                                        lcoe_thresh=lcoe_thresh,
                                        offshore=offshore,
                                        max_workers=max_workers,
                                        plants_per_worker=plants_per_worker,
                                        points_per_worker=points_per_worker)
        else:
            self._plants = Plants.load(plants)

        self._sc_bus_dist = None

    def __repr__(self):
        msg = "{} with {} plants".format(self.__class__.__name__, len(self))
        return msg

    def __len__(self):
        return len(self.plexos_table)

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
    def mymean_fpath(self):
        """
        reV multi year mean output file path

        Returns
        -------
        str
        """
        return self._mymean_fpath

    @property
    def cf_fpath(self):
        """
        reV generation output file path

        Returns
        -------
        str
        """
        return self._cf_fpath

    @property
    def cf_gid_map(self):
        """Mapping of res_gid (index) to gen_gid (values)

        Returns
        -------
        pd.Series
        """
        return self._cf_gid_map

    @property
    def sc_table(self):
        """
        Supply Curve Table

        Returns
        -------
        pandas.DataFrame
        """
        return self._sc_table

    @property
    def plants(self):
        """
        Dictionary matching plants to plant ids

        Returns
        -------
        dict
        """
        return self._plants.plants

    @property
    def plant_builds(self):
        """
        PLEXOS Plant builds

        Returns
        -------
        dict
        """
        return self._plants.plant_builds

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

            self._sc_bus_dist = \
                PlexosPlants._haversine_dist(plant_coords, sc_coords).T

        return self._sc_bus_dist

    @property
    def unique_plant_names(self):
        """Get a list of ordered unique plant names if plant_name_col
        was provided.

        Returns
        -------
        list | None
        """
        names = None

        if self._plant_name_col is not None:
            names = BaseProfileAggregation.get_unique_plant_names(
                self.plexos_table, self._plant_name_col, self._tech_tag)

        return names

    @property
    def tz_alias(self):
        """Get a short 3-char tz alias if the timezone is common in the US
        (pst, mst, cst, est)

        Returns
        -------
        str
        """
        return TZ_ALIASES.get(self._timezone, self._timezone)

    def get_gen_gid(self, res_gid):
        """Get a generation gid from a resource gid using cf_gid_map. Accounts
        for a many-to-one resource-to-gen_gid mapping.

        Parameters
        ----------
        res_gid : int

        Returns
        -------
        gen_gid
        """

        res_gid = self.cf_gid_map.loc[res_gid]

        if isinstance(res_gid, pd.Series):
            res_gid = res_gid.values[0]

        return res_gid

    @staticmethod
    def _parse_plexos_table(plexos_table, plant_name_col=None):
        """
        Parse PLEXOS table from file and reduce to PLEXOS_COLS
        Combine buses at the same coordinates and add unique plant_ids

        Parameters
        ----------
        plexos_table : str | pandas.DataFrame
            PLEXOS table of bus locations and capacity (MW) provided as a .csv,
            .json, or pandas DataFrame
        plant_name_col : str | None
            Column in plexos_table that has the plant name that should be used
            in the plexos output csv column headers.

        Returns
        -------
        plexos_table : pandas.DataFrame
            Parsed and clean PLEXOS table
        """
        logger.info('Parsing plexos table')
        plexos_table = parse_table(plexos_table)
        cols = ['generator', 'busid', 'busname', 'capacity', 'latitude',
                'longitude', 'system']

        rename = {c: c.lower() for c in plexos_table}
        plexos_table = plexos_table.rename(columns=rename)

        if plant_name_col is not None:
            msg = ('plant_name_col "{}" not in plexos table!'
                   .format(plant_name_col))
            assert plant_name_col.lower() in plexos_table, msg
            cols.append(plant_name_col.lower())

        cols = [c.lower() for c in plexos_table if c.lower() in cols]
        plexos_table = plexos_table[cols]

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
        plexos_table = plexos_table.loc[mask]
        cols = ['latitude', 'longitude']
        plant_cap = plexos_table.groupby(cols)['capacity'].sum()
        plant_cap = plant_cap.reset_index().reset_index()
        rename = {'index': 'plant_id', 'capacity': 'plant_capacity'}
        plant_cap = plant_cap.rename(columns=rename)

        plexos_table = plexos_table.merge(plant_cap,
                                          on=cols,
                                          how='inner')

        return plexos_table

    @staticmethod
    def _parse_cf_gid_map(cf_fpath):
        """
        Map resource gids to gen gids

        Parameters
        ----------
        cf_fpath : str
            Path to reV generation output .h5 file

        Returns
        -------
        cf_gid_map : pd.Series
            Mapping of res_gid (index) to gen_gid (values)
        """
        logger.info('Mapping reV resource GIDs to generation GIDs.')
        with Resource(cf_fpath) as f:
            res_gids = f.get_meta_arr('gid')

        if not isinstance(res_gids, np.ndarray):
            res_gids = np.array(list(res_gids))

        cf_gid_map = pd.Series(np.arange(len(res_gids)), index=res_gids,
                               name='gen_gid')
        cf_gid_map.index.name = 'res_gid'

        return cf_gid_map

    @staticmethod
    def _collapse_multi_list(multi_list):
        """Collapse a list of lists into one list"""
        if isinstance(multi_list, (pd.Series, pd.DataFrame)):
            multi_list = multi_list.values.tolist()

        if any(isinstance(x, (list, tuple)) for x in multi_list):
            multi_list = [item for sublist in multi_list for item in sublist]

        return multi_list

    def plants_meta(self):
        """Create plants meta data from filled plants DataFrames:

            - Location (lat, lon)
            - final capacity (MW)
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
        for pid, plant in self.plant_builds.items():
            single_meta = {
                'sc_gids': plant['sc_gid'].values.tolist(),
                'res_gids': plant['res_gids'].values.tolist(),
                'gid_counts': plant['gid_counts'].values.tolist(),
                'gen_gids': [[self.get_gen_gid(gid) for gid in gids]
                             for gids in plant['res_gids'].values],
                'res_cf_means': plant['cf_means'].values.tolist(),
                'build_capacity': plant['build_capacity'].values.tolist()}
            single_meta = {k: self._collapse_multi_list(v)
                           for k, v in single_meta.items()}
            single_meta['cf_mean'] = np.hstack(plant['cf_means'].values).mean()
            plants_meta.append(pd.Series(single_meta, name=pid))

        plants_meta = pd.concat(plants_meta, axis=1).T
        plants_meta.index.name = 'plant_id'

        plants_meta = self.plexos_table.merge(plants_meta.reset_index(),
                                              on='plant_id', how='outer')

        return plants_meta

    def _make_plant_meta(self, bus_meta):
        """
        Create plant meta data for given bus

        Parameters
        ----------
        bus_meta : pandas.Series
            Meta data for desired bus to build plant for

        Returns
        -------
        plant_meta : pandas.DataFrame
            Meta data for plant associated with given bus, constructed from:
            - Plant table
            - Supply Curve table
            - Bus capacity (MW)
        """
        plant_meta = self.plant_builds[bus_meta['plant_id']]
        plant_meta['gen_gids'] = \
            plant_meta['res_gids'].apply(lambda gids: [self.get_gen_gid(gid)
                                                       for gid in gids])

        sc_cols = ['res_gids', 'gen_gids', 'gid_counts', 'capacity']
        sc_cols = [c for c in self.sc_table if c not in sc_cols]
        plant_meta = plant_meta.merge(self.sc_table[sc_cols],
                                      on='sc_gid', how='left')

        plant_capacity = plant_meta['build_capacity'].sum()
        if plant_capacity != bus_meta['capacity']:
            bulid_capacity = (plant_meta['build_capacity'] / plant_capacity
                              * bus_meta['capacity'])
            plant_meta.loc[:, 'build_capacity'] = bulid_capacity

        return plant_meta

    @staticmethod
    def _make_profile(cf_fpath, plant_build, dset_tag=''):
        """
        Make generation profiles for given plant buildout

        Parameters
        ----------
        cf_fpath : str
            Path to reV Generation output .h5 file to pull CF profiles from
        plant_build : pandas.DataFrame
            DataFrame describing plant buildout:
                - Supply curve gids and the capacity (MW) to build at each
                    - res_gids, gen_gids, gid_counts by sc_gid
        dset_tag : str
            Dataset tag to append to dataset names in cf profile file. e.g. If
            the cf profile file is a multi year file using dset_tag="-2008"
            will enable us to select the corresponding datasets
            (cf_mean-2008, cf_profile-2008, etc)

        Returns
        -------
        profile: ndarray
            Generation profile for plant as a vector
        """
        with Resource(cf_fpath) as f:
            profile = None
            for _, row in plant_build.iterrows():
                gid_capacities = (row['gid_counts'] / np.sum(row['gid_counts'])
                                  * row['build_capacity'])
                cf_profiles = f['cf_profile' + dset_tag, :, row['gen_gids']]
                for i, cf_profile in enumerate(cf_profiles.T):
                    if profile is None:
                        profile = cf_profile * gid_capacities[i]
                    else:
                        profile += cf_profile * gid_capacities[i]

        if len(profile.shape) != 1:
            profile = profile.flatten()

        return profile

    def aggregate_profiles(self, out_fpath):
        """
        Aggregate plants from capacity factor profiles and save to given
        output .h5 path

        Parameters
        ----------
        out_fpath : str
            .h5 path to save aggregated plant profiles to
            A companion .csv with be saved at the same location for plexos.
        """

        if not out_fpath.endswith('.h5'):
            out_fpath = out_fpath + '.h5'

        if not os.path.exists(os.path.dirname(out_fpath)):
            os.makedirs(os.path.dirname(out_fpath))

        out_fpath = out_fpath.replace('.h5', f'_{self.tz_alias}.h5')

        with Outputs(out_fpath, mode='w') as f_out:
            f_out.set_version_attr()
            with Resource(self.cf_fpath) as f_in:
                logger.info('Copying time_index')
                ti = f_in['time_index' + self._dset_tag].tz_convert(None)
                f_out['time_index'] = ti

            logger.info('Writing meta data')
            f_out['meta'] = self.plants_meta()

            f_out.h5.create_group('plant_meta')
            gen_profiles = []
            logger.info('Extracting profiles and writing meta for plants')
            for i, irow in enumerate(self.plexos_table.iterrows()):
                bus_id, bus_meta = irow
                logger.debug('Building plant for bus {}'.format(bus_id))
                plant_meta = self._make_plant_meta(bus_meta)
                prof = self._make_profile(self.cf_fpath, plant_meta.copy(),
                                          dset_tag=self._dset_tag)
                gen_profiles.append(prof)

                plant_meta = to_records_array(plant_meta)
                logger.debug('Writing plant_meta/{} ({} out of {})'
                             .format(bus_id, i + 1, len(self.plexos_table)))
                f_out._create_dset('plant_meta/{}'.format(bus_id),
                                   plant_meta.shape,
                                   plant_meta.dtype,
                                   chunks=None,
                                   data=plant_meta)

            logger.info('Writing Generation Profiles')
            gen_profiles = np.dstack(gen_profiles)[0].astype('float32')
            gen_profiles = BaseProfileAggregation.tz_convert_profiles(
                gen_profiles, self._timezone)
            f_out._create_dset('gen_profiles',
                               gen_profiles.shape,
                               gen_profiles.dtype,
                               chunks=(None, 100),
                               data=gen_profiles,
                               attrs={'units': 'MW'})

        logger.info('Finished aggregating profiles to: {}'.format(out_fpath))

        df_plx = pd.DataFrame(gen_profiles, columns=self.unique_plant_names,
                              index=ti)
        df_plx.index.name = 'DATETIME'
        csv_fp = out_fpath.replace('.h5', '.csv')
        df_plx.to_csv(csv_fp)

        logger.info('Wrote plexos formatted profiles to: {}'.format(csv_fp))

    @classmethod
    def aggregate(cls, plexos_table, sc_table, mymean_fpath, cf_fpath,
                  plants_fpath, out_fpath, offshore=False):
        """
        Aggregate pre-filled plants

        Parameters
        ----------
        plexos_table : str | pandas.DataFrame
            PLEXOS table of bus locations and capacity (MW) provided as a .csv,
            .json, or pandas DataFrame
        sc_table : str | pandas.DataFrame
            Supply Curve table .csv or pre-loaded pandas DataFrame
        mymean_fpath : str
            Path to reV multi-year-mean output .h5 file to pull cf_mean-means
            from - this will be used to determine where plants are built so it
            does not differ from year to year. The meta must match cf_fpath.
        cf_fpath : str
            Path to reV annual Generation output .h5 file to pull CF profiles
            from. Meta must match mymean_fpath.
        plants_fpath : str
            Path to .csv containing pre-filled plants
        out_fpath : str
            .h5 path to save aggregated plant profiles to
        offshore : bool, optional
            Include offshore points, by default False
        """
        pp = cls(plexos_table, sc_table, mymean_fpath, cf_fpath,
                 plants=plants_fpath, offshore=offshore)
        # Add plants to PlexosPlant instance

        pp.aggregate_profiles(out_fpath)

    @classmethod
    def run(cls, plexos_table, sc_table, mymean_fpath, cf_fpath, out_fpath,
            dist_percentile=90, dist_thresh_km=None,
            lcoe_col='total_lcoe', lcoe_thresh=1.3,
            max_workers=None, points_per_worker=400, plants_per_worker=40,
            offshore=False, plant_name_col=None, tech_tag=None, dset_tag='',
            timezone='UTC', bespoke=False):
        """
        Find, fill, and save profiles for Plants associated with given PLEXOS
        buses

        Parameters
        ----------
        plexos_table : str | pandas.DataFrame
            PLEXOS table of bus locations and capacity (MW) provided as a .csv,
            .json, or pandas DataFrame. Needs columns: latitude, longitude,
            capacity. Note that capacity needs to be AC for wind and DC for
            solar.
        sc_table : str | pandas.DataFrame
            Supply Curve table .csv or pre-loaded pandas DataFrame
        mymean_fpath : str
            Path to reV multi-year-mean output .h5 file to pull cf_mean-means
            from - this will be used to determine where plants are built so it
            does not differ from year to year. The meta must match cf_fpath.
        cf_fpath : str
            Path to reV annual Generation output .h5 file to pull CF profiles
            from. Meta must match mymean_fpath.
        out_fpath : str
            .h5 path to save aggregated plant profiles to
            A companion .csv with be saved at the same location for plexos.
        dist_percentile : int, optional
            Percentile to use to compute distance threshold using sc_gid to
            SubStation distance , by default 90
        dist_thresh_km : float | None
            Optional absolute distance threshold in km that will override the
            dist_percentile input.
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
        plant_name_col : str | None
            Column in plexos_table that has the plant name that should be used
            in the plexos output csv column headers.
        tech_tag : str | None
            Optional technology tag to include as a suffix in the plexos output
            csv column headers.
        dset_tag : str
            Dataset tag to append to dataset names in cf profile file. e.g. If
            the cf profile file is a multi year file using dset_tag="-2008"
            will enable us to select the corresponding datasets
            (cf_mean-2008, cf_profile-2008, etc)
        timezone : str
            Timezone for output generation profiles. This is a string that will
            be passed to pytz.timezone() e.g. US/Pacific, US/Mountain,
            US/Central, US/Eastern, or UTC. For a list of all available
            timezones, see pytz.all_timezones
        bespoke : bool
            Flag to signify if the cf_fpath file was generated using the reV
            bespoke wind module. The bespoke output files have generation
            profiles at the supply curve grid resolution which is different
            than traditional reV generation outputs that are on the resource
            grid resolution.
        """
        pp = cls(plexos_table, sc_table, mymean_fpath, cf_fpath,
                 offshore=offshore,
                 dist_percentile=dist_percentile,
                 dist_thresh_km=dist_thresh_km,
                 lcoe_col=lcoe_col,
                 lcoe_thresh=lcoe_thresh,
                 max_workers=max_workers,
                 points_per_worker=points_per_worker,
                 plants_per_worker=plants_per_worker,
                 plant_name_col=plant_name_col,
                 tech_tag=tech_tag,
                 dset_tag=dset_tag,
                 timezone=timezone,
                 bespoke=bespoke)

        pp.aggregate_profiles(out_fpath)
