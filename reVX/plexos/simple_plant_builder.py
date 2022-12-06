# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:47:43 2019

@author: gbuster
"""
from concurrent.futures import as_completed
import json
import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from rex.utilities.execution import SpawnProcessPool
from rex.utilities.utilities import parse_table

from reVX.plexos.base import BaseProfileAggregation, PlexosNode
from reVX.plexos.utilities import get_coord_labels

logger = logging.getLogger(__name__)


class SimplePlantBuilder(BaseProfileAggregation):
    """Class to build generation profiles for "plants" by aggregating resource
    from nearest neighbor supply curve points.
    """

    def __init__(self, plant_meta, rev_sc, cf_fpath, forecast_fpath=None,
                 plant_name_col=None, tech_tag=None, timezone='UTC',
                 share_resource=True, max_workers=None):
        """Run plexos aggregation.

        Parameters
        ----------
        plant_meta : str | pd.DataFrame
            Str filepath or extracted dataframe for plant meta data with every
            row representing a plant with columns for latitude, longitude,
            and capacity (in MW). Plants will compete for available capacity
            in the reV supply curve input and will be prioritized based on the
            row order of this input.
        rev_sc : str | pd.DataFrame
            reV supply curve or sc-aggregation output table including sc_gid,
            latitude, longitude, res_gids, gid_counts, mean_cf.
        cf_fpath : str
            File path to capacity factor file (reV gen output) to
            get profiles from.
        forecast_fpath : str | None
            Forecasted capacity factor .h5 file path (reV results).
            If not None, the generation profiles are sourced from this file.
        plant_name_col : str | None
            Column in plexos_table that has the plant name that should be used
            in the plexos output csv column headers.
        tech_tag : str | None
            Optional technology tag to include as a suffix in the plexos output
            csv column headers.
        timezone : str
            Timezone for output generation profiles. This is a string that will
            be passed to pytz.timezone() e.g. US/Pacific, US/Mountain,
            US/Central, US/Eastern, or UTC. For a list of all available
            timezones, see pytz.all_timezones
        share_resource : bool
            Flag to share available capacity within a single resource GID
            between multiple plants.
        max_workers : int | None
            Max workers for parallel profile aggregation. None uses all
            available workers. 1 will run in serial.
        """
        logger.info('Initializing SimplePlantBuilder.')
        super().__init__()
        self._res_gids = None
        self._plant_meta = parse_table(plant_meta).reset_index(drop=True)
        self._sc_table = parse_table(rev_sc).reset_index(drop=True)
        self._cf_fpath = cf_fpath
        self._forecast_fpath = forecast_fpath
        self._share_res = share_resource
        self._output_meta = None
        self._plant_name_col = plant_name_col
        self._tech_tag = tech_tag
        self._timezone = timezone
        self.max_workers = max_workers

        required = ('sc_gid', 'latitude', 'longitude', 'res_gids',
                    'gid_counts', 'mean_cf')
        missing = [r not in self._sc_table for r in required]
        if any(missing):
            msg = ('SimplePlantBuilder needs the following missing columns '
                   'in the rev_sc input: {}'.format(missing))
            logger.error(msg)
            raise ValueError(msg)

        required = ('latitude', 'longitude', 'capacity')
        missing = [r not in self._plant_meta for r in required]
        if any(missing):
            msg = ('SimplePlantBuilder needs the following missing columns '
                   'in the plant_meta input: {}'.format(missing))
            logger.error(msg)
            raise ValueError(msg)

        self._node_map = self._make_node_map()
        self._forecast_map = self._make_forecast_map(self._cf_fpath,
                                                     self._forecast_fpath)
        self._compute_gid_capacities()
        logger.info('Finished initializing SimplePlantBuilder.')

    def _compute_gid_capacities(self):
        """Compute the individual resource gid capacities and make a new
        column in the SC table."""

        for label in ('res_gids', 'gid_counts'):
            if isinstance(self._sc_table[label].values[0], str):
                self._sc_table[label] = self._sc_table[label].apply(json.loads)

        self._sc_table['gid_capacity'] = None
        for i, row in self._sc_table.iterrows():
            gid_counts = row['gid_counts']
            gid_capacity = gid_counts / np.sum(gid_counts) * row['capacity']
            self._sc_table.at[i, 'gid_capacity'] = list(gid_capacity)

    def _make_node_map(self):
        """Run haversine balltree to map plant locations to nearest supply
        supply curve points

        Returns
        -------
        ind : np.ndarray
            BallTree (haversine) query output, (n, m) array of plant indices
            mapped to the SC points where n is the number of plants, m is the
            number of SC points, and each row in the array yields the sc points
            m closest to the plant n.
        """
        logger.debug('Making node map...')

        plant_coord_labels = get_coord_labels(self._plant_meta)
        sc_coord_labels = get_coord_labels(self._sc_table)

        # pylint: disable=not-callable
        sc_coords = np.radians(self._sc_table[sc_coord_labels].values)
        plant_coords = np.radians(self._plant_meta[plant_coord_labels])
        tree = BallTree(sc_coords, metric='haversine')
        ind = tree.query(plant_coords, return_distance=False,
                         k=len(self._sc_table))
        logger.debug('Finished mkaing node map.')

        return ind

    @property
    def plant_meta(self):
        """Get plant meta data for the requested plant buildout
        with buildout information

        Returns
        -------
        pd.DataFrame
        """

        if self._output_meta is None:
            self._output_meta = self._plant_meta.copy()

            self._output_meta['sc_gids'] = None
            self._output_meta['res_gids'] = None
            self._output_meta['gen_gids'] = None
            self._output_meta['res_built'] = None

        return self._output_meta

    def assign_plant_buildouts(self):
        """March through the plant meta data and make subsets of the supply
        curve table that will be built out for each plant. The supply curve
        table attribute of this SimplePlantBuilder instance will be manipulated
        such that total sc point capacity and resource gid capacity is reduced
        whenever a plant is built. In this fashion, resource in SC points will
        not be double counted, but resource within an SC point can be divided
        up between multiple plants. Resource within an SC point is prioritized
        by available capacity.

        Returns
        -------
        plant_sc_builds : dict
            Dictionary mapping the plant row indices (keys) to subsets of the
            SC table showing what should be built for each plant. The subset
            SC tables in this dict will no longer match the sc table attribute
            of the SimplePlantBuilder instance, because the tables in this dict
            show what should be built, and the sc table attribute will show
            what is remaining.
        """

        plant_sc_builds = {}
        built_res_gids = []

        # March through plant meta data table in order provided
        for i, plant_row in self._plant_meta.iterrows():
            logger.debug('Starting plant buildout assignment for plant {} '
                         'out of {}'.format(i + 1, len(self._plant_meta)))

            plant_cap_to_build = float(plant_row['capacity'])
            single_plant_sc = pd.DataFrame()

            # March through the SC table in order of the node map
            for sc_loc in self.node_map[i]:
                sc_point = self._sc_table.loc[sc_loc].copy()
                sc_capacity = sc_point['capacity']

                # Buildout capacity in this sc point
                if sc_capacity >= 0:
                    # Make arrays of gid capacities that will be built
                    # for this plant and also saved for subsequent plants.
                    sc_point_res_gids = np.array(sc_point['res_gids'])
                    cap_orig = np.array(sc_point['gid_capacity'])
                    cap_remain = cap_orig.copy()
                    cap_build = np.zeros_like(cap_orig)

                    # Build greatest available capacity first
                    order = np.flip(np.argsort(cap_orig))

                    for j in order:
                        res_gid = sc_point_res_gids[j]
                        if self._share_res or res_gid not in built_res_gids:
                            # add built capacity to the "to build" array
                            # (on a resource point per sc point basis)
                            # and remove from the "remaining" array
                            built = np.minimum(plant_cap_to_build, cap_orig[j])
                            cap_build[j] += built
                            cap_remain[j] -= built
                            plant_cap_to_build -= built
                            built_res_gids.append(res_gid)

                        # buildout for this plant is fully complete
                        if plant_cap_to_build <= 0:
                            break

                    if cap_build.sum() > 0:
                        # Capacity was built in this SC point
                        cap_build = cap_build.tolist()
                        cap_orig = cap_orig.tolist()

                        sc_point['capacity'] = sum(cap_build)
                        sc_point['built_capacity'] = sum(cap_build)
                        sc_point['gid_capacity'] = cap_build
                        single_plant_sc = single_plant_sc.append(sc_point)

                        self._sc_table.at[sc_loc, 'capacity'] -= sum(cap_build)
                        self._sc_table.at[sc_loc, 'gid_capacity'] = cap_remain

                # buildout for this plant is fully complete
                if plant_cap_to_build <= 0:
                    plant_sc_builds[i] = single_plant_sc
                    break

        logger.info('Finished plant buildout assignment.')

        return plant_sc_builds

    def check_valid_buildouts(self, plant_sc_builds):
        """Check that plant buildouts are mapped to valid resource data that
        can be found in the cf_fpath input."""

        global_built_res_gids = []
        for i, single_plant_sc in plant_sc_builds.items():
            sc_res_gids = single_plant_sc['res_gids'].values.tolist()
            sc_res_gids = [g for subset in sc_res_gids for g in subset]
            gid_caps = single_plant_sc['gid_capacity'].values.tolist()
            gid_caps = [g for subset in gid_caps for g in subset]
            assert len(gid_caps) == len(sc_res_gids)
            plant_built_res_gids = [gid for j, gid in enumerate(sc_res_gids)
                                    if gid_caps[j] > 0]
            missing = [gid for gid in plant_built_res_gids
                       if gid not in self.available_res_gids]
            if any(missing):
                msg = ('Plant index {} was mapped to resource gids that are '
                       'missing from the cf file: {}'.format(i, missing))
                logger.error(msg)
                raise RuntimeError(msg)

            shared = [gid for gid in plant_built_res_gids
                      if gid in global_built_res_gids]
            if any(shared) and not self._share_res:
                msg = ('SimplePlantBuilder shared resource gids when it '
                       'should not have: {}'.format(shared))
                logger.error(msg)
                raise RuntimeError(msg)
            else:
                global_built_res_gids += plant_built_res_gids

    def make_profiles(self, plant_sc_builds):
        """Make a 2D array of aggregated plant gen profiles.

        Returns
        -------
        profiles : np.ndarray
            (t, n) array of plant  eneration profiles where t is the
            timeseries length and n is the number of plants.
        """

        if self.max_workers != 1:
            profiles = self._make_profiles_parallel(plant_sc_builds)
        else:
            profiles = self._make_profiles_serial(plant_sc_builds)

        return profiles

    def _make_profiles_parallel(self, plant_sc_builds):
        """Make a 2D array of aggregated plant gen profiles in parallel.

        Returns
        -------
        profiles : np.ndarray
            (t, n) array of plant node generation profiles where t is the
            timeseries length and n is the number of plants.
        """

        logger.info('Starting plant profile buildout in parallel.')
        profiles = self._init_output(len(self.plant_meta))
        progress = 0
        futures = {}
        loggers = [__name__, 'reVX']
        with SpawnProcessPool(max_workers=self.max_workers,
                              loggers=loggers) as exe:
            for i, plant_sc_subset in plant_sc_builds.items():
                f = exe.submit(PlexosNode.run,
                               plant_sc_subset, self._cf_fpath,
                               res_gids=self.available_res_gids,
                               forecast_fpath=self._forecast_fpath,
                               forecast_map=self._forecast_map)
                futures[f] = i

            for n, f in enumerate(as_completed(futures)):
                i = futures[f]
                profile, sc_gids, res_gids, gen_gids, res_built = f.result()
                profiles[:, i] = profile
                self._ammend_output_meta(i, sc_gids, res_gids, gen_gids,
                                         res_built)

                current_prog = (n + 1) // (len(futures) / 100)
                if current_prog > progress:
                    progress = current_prog
                    logger.info('{} % of plant node profiles built.'
                                .format(progress))

        logger.info('Finished plant profile buildout.')
        return profiles

    def _make_profiles_serial(self, plant_sc_builds):
        """Make a 2D array of aggregated plexos gen profiles in serial.

        Returns
        -------
        profiles : np.ndarray
            (t, n) array of Plexos node generation profiles where t is the
            timeseries length and n is the number of plexos nodes.
        """

        logger.info('Starting plant profile buildout in serial.')
        profiles = self._init_output(len(self.plant_meta))
        progress = 0
        for i, plant_sc_subset in plant_sc_builds.items():
            p = PlexosNode.run(
                plant_sc_subset, self._cf_fpath,
                res_gids=self.available_res_gids,
                forecast_fpath=self._forecast_fpath,
                forecast_map=self._forecast_map)

            profile, sc_gids, res_gids, gen_gids, res_built = p
            profiles[:, i] = profile
            self._ammend_output_meta(i, sc_gids, res_gids, gen_gids, res_built)

            current_prog = ((i + 1)
                            // (len(np.unique(self.node_map)) / 100))
            if current_prog > progress:
                progress = current_prog
                logger.info('{} % of plant profiles built.'
                            .format(progress))

        logger.info('Finished plant profile buildout.')
        return profiles

    @classmethod
    def run(cls, plant_meta, rev_sc, cf_fpath, forecast_fpath=None,
            plant_name_col=None, tech_tag=None, timezone='UTC',
            share_resource=True, max_workers=None, out_fpath=None):
        """Build profiles and meta data.

        Parameters
        ----------
        plant_meta : str | pd.DataFrame
            Str filepath or extracted dataframe for plant meta data with every
            row representing a plant with columns for latitude, longitude,
            and capacity (in MW). Plants will compete for available capacity
            in the reV supply curve input and will be prioritized based on the
            row order of this input.
        rev_sc : str | pd.DataFrame
            reV supply curve or sc-aggregation output table including sc_gid,
            latitude, longitude, res_gids, gid_counts, mean_cf.
        cf_fpath : str
            File path to capacity factor file (reV gen output) to
            get profiles from.
        forecast_fpath : str | None
            Forecasted capacity factor .h5 file path (reV results).
            If not None, the generation profiles are sourced from this file.
        plant_name_col : str | None
            Column in plexos_table that has the plant name that should be used
            in the plexos output csv column headers.
        tech_tag : str | None
            Optional technology tag to include as a suffix in the plexos output
            csv column headers.
        timezone : str
            Timezone for output generation profiles. This is a string that will
            be passed to pytz.timezone() e.g. US/Pacific, US/Mountain,
            US/Central, US/Eastern, or UTC. For a list of all available
            timezones, see pytz.all_timezones
        share_resource : bool
            Flag to share available capacity within a single resource GID
            between multiple plants.
        max_workers : int | None
            Max workers for parallel profile aggregation. None uses all
            available workers. 1 will run in serial.
        out_fpath : str, optional
            Path to .h5 file into which plant buildout should be saved. A
            plexos-formatted csv will also be written in the same directory.
            By default None.

        Returns
        -------
        plant_meta : pd.DataFrame
            Plant meta data with built capacities and mappings to the
            resource used.
        time_index : pd.datetimeindex
            Time index for the profiles.
        profiles : np.ndarray
            Generation profile timeseries in MW at each plant.
        """

        pb = cls(plant_meta, rev_sc, cf_fpath, forecast_fpath=forecast_fpath,
                 plant_name_col=plant_name_col, tech_tag=tech_tag,
                 timezone=timezone, share_resource=share_resource,
                 max_workers=max_workers)

        plant_sc_builds = pb.assign_plant_buildouts()
        pb.check_valid_buildouts(plant_sc_builds)
        profiles = pb.make_profiles(plant_sc_builds)

        if out_fpath is not None:
            pb.export(pb.plant_meta, pb.time_index, profiles, out_fpath)

        return pb.plant_meta, pb.time_index, profiles
