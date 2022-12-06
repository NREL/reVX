# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:47:43 2019

@author: gbuster
"""
from concurrent.futures import as_completed
import json
import logging
import numpy as np
import os
import pandas as pd
from scipy.spatial import cKDTree
from geopandas import GeoDataFrame
from warnings import warn

from rex.utilities.execution import SpawnProcessPool
from rex.utilities.utilities import parse_table, to_records_array

from reVX.utilities.region_classifier import RegionClassifier
from reVX.handlers.outputs import Outputs
from reVX.plexos.base import BaseProfileAggregation, PlexosNode
from reVX.plexos.utilities import (DataCleaner, get_coord_labels,
                                   parse_table_name)

logger = logging.getLogger(__name__)


class PlexosAggregation(BaseProfileAggregation):
    """
    Framework to aggregate reV gen profiles to PLEXOS node power profiles.
    This class takes as input the plexos nodes meta data (lat/lon or shape
    files), rev supply curve table, and reeds buildout table (specifying
    which rev sc points were built and at what capacity). The class
    will build power profiles for each supply curve point and then aggregate
    the sc point profiles to the nearest neighbor plexos node (if plexos nodes
    are defined by lat/lon) or the shape intersect plexos node (if plexos nodes
    are defined by shape file).
    """

    def __init__(self, plexos_nodes, rev_sc, reeds_build, cf_fpath,
                 forecast_fpath=None, build_year=2050, plexos_columns=None,
                 force_full_build=False, force_shape_map=False,
                 plant_name_col=None, tech_tag=None, timezone='UTC',
                 max_workers=None):
        """
        Parameters
        ----------
        plexos_nodes : str | pd.DataFrame
            Plexos node meta data including gid, latitude, longitude, voltage.
            Or file path to .csv containing plexos node meta data, or a file
            path to a .shp file that contains plexos nodes defined as shapes.
        rev_sc : str | pd.DataFrame
            reV supply curve results table including SC gid, latitude,
            longitude, res_gids, gid_counts. Or file path to reV supply
            curve table.
        reeds_build : str | pd.DataFrame
            ReEDS buildout with rows for built capacity (MW) at each reV SC
            point. This should have columns: reeds_year, built_capacity, and
            sc_gid (corresponding to the reV supply curve point gid). Some
            cleaning of the column names will be performed for legacy tables
            but these are the column headers that are desired. This input can
            also include "plexos_node_gid" which will explicitly assign a
            supply curve point buildout to a single plexos node. If included,
            all points must be assigned to plexos nodes.
        cf_fpath : str
            File path to capacity factor file (reV gen output) to
            get profiles from.
        forecast_fpath : str | None
            Forecasted capacity factor .h5 file path (reV results). If not
            None, the supply curve res_gids are mapped to sites in the
            cf_fpath, then the coordinates from cf_fpath are mapped to the
            nearest neighbor sites in the forecast_fpath, where the final
            generation profiles are retrieved from.
        build_year : int, optional
            REEDS year of interest, by default 2050
        plexos_columns : list | None
            Additional columns from the plexos_nodes input to pass through
            to the output meta data.
        force_full_build : bool
            Flag to ensure the full requested buildout is built at each SC
            point. If True, the remainder of the requested build will always
            be built at the last resource gid in the sc point.
        force_shape_map : bool
            Flag to force the mapping of supply curve points to the plexos
            node shape file input (if a shape file is input) via nearest
            neighbor to shape centroid.
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
        max_workers : int | None
            Max workers for parallel profile aggregation. None uses all
            available workers. 1 will run in serial.
        """

        super().__init__()
        self._cf_fpath = cf_fpath
        self._forecast_fpath = forecast_fpath
        self.build_year = build_year
        self._res_gids = None
        self._output_meta = None
        self._time_index = None
        self._force_full_build = force_full_build
        self._force_shape_map = force_shape_map
        self.max_workers = max_workers
        self._plant_name_col = plant_name_col
        self._tech_tag = tech_tag
        self._timezone = timezone

        if plexos_columns is None:
            plexos_columns = tuple()
        self._plexos_columns = plexos_columns
        self._plexos_columns += DataCleaner.PLEXOS_META_COLS
        self._plexos_columns = tuple(set(self._plexos_columns))

        logger.info('Running PLEXOS aggregation for build year: {}'
                    .format(build_year))

        self._sc_build = self._parse_rev_reeds(rev_sc, reeds_build,
                                               build_year=build_year)
        self._plexos_nodes = self._parse_plexos_nodes(plexos_nodes)

        missing = self._check_gids()
        self._handle_missing_resource_gids(missing)

        self._node_map = self._make_node_map()
        self._forecast_map = self._make_forecast_map(self._cf_fpath,
                                                     self._forecast_fpath)

    @property
    def plexos_meta(self):
        """Get plexos node meta data for the nodes included in this problem.

        Returns
        -------
        plexos_meta : pd.DataFrame
            Plexos meta dataframe reduced to the nodes in this problem.
        """

        if self._output_meta is None:
            inodes = np.unique(self.node_map)

            node_builds = []
            for i in inodes:
                mask = (self.node_map == i)
                built_cap = self.sc_build[mask]['built_capacity'].values.sum()
                node_builds.append(built_cap)

            self._output_meta = self._plexos_nodes.iloc[inodes, :]
            self._output_meta['built_capacity'] = node_builds

            self._output_meta = DataCleaner.reduce_df(
                self._output_meta, self._plexos_columns)

            self._output_meta['sc_gids'] = None
            self._output_meta['res_gids'] = None
            self._output_meta['gen_gids'] = None
            self._output_meta['res_built'] = None

        return self._output_meta

    @property
    def n_plexos_nodes(self):
        """Get the number of unique plexos nodes in this buildout.

        Returns
        -------
        n : int
            Number of unique plexos nodes in this buildout
        """
        return len(self.plexos_meta)

    @property
    def sc_res_gids(self):
        """List of unique resource GIDS in the REEDS build out.

        Returns
        -------
        sc_res_gids : np.ndarray
            Array of resource GIDs associated with this REEDS buildout.
        """

        gid_col = self.sc_build['res_gids'].values

        if isinstance(gid_col[0], str):
            gid_col = [json.loads(s) for s in gid_col]
        else:
            gid_col = list(gid_col)

        res_gids = [g for sub in gid_col for g in sub]
        sc_res_gids = np.array(sorted(list(set(res_gids))))

        return sc_res_gids

    @property
    def sc_build(self):
        """Get the reV supply curve table reduced to just those points built
        by reeds including a built_capacity column in MW.

        Returns
        -------
        pd.DataFrame
        """
        return self._sc_build

    def _parse_plexos_nodes(self, plexos_nodes):
        """
        Load Plexos node meta data from disc if needed, pre-filter and rename
        columns

        Parameters
        ----------
        plexos_nodes : str | pd.DataFrame
            Plexos node meta data including gid, latitude, longitude, voltage.
            Or file path to .csv containing plexos node meta data, or a file
            path to a .shp file that contains plexos nodes defined as shapes.

        Returns
        -------
        plexos_nodes : pd.DataFrame
            Plexos node meta data including gid, latitude, longitude, voltage
        """

        if (isinstance(plexos_nodes, str)
                and plexos_nodes.endswith(('.csv', '.json'))):
            plexos_nodes = parse_table(plexos_nodes)

        elif isinstance(plexos_nodes, str) and plexos_nodes.endswith('.shp'):
            rc = RegionClassifier(self.sc_build, plexos_nodes,
                                  regions_label=None)
            plexos_nodes = rc._regions
            if 'plexos_id' not in plexos_nodes:
                plexos_nodes['plexos_id'] = np.arange(len(plexos_nodes))

        elif not isinstance(plexos_nodes, pd.DataFrame):
            msg = ('Expected a DataFrame or a file path to csv, json, or '
                   'shp for the plexos_nodes input but received: {} ({})'
                   .format(plexos_nodes, type(plexos_nodes)))
            logger.error(msg)
            raise NotImplementedError(msg)

        plexos_nodes = DataCleaner.rename_cols(plexos_nodes)
        plexos_nodes = DataCleaner.pre_filter_plexos_meta(plexos_nodes)

        return plexos_nodes

    @staticmethod
    def _check_rev_reeds_coordinates(rev_sc, reeds_build, atol=0.5):
        """Check that the coordinates are the same in rev and reeds buildouts.

        Parameters
        ----------
        rev_sc : pd.DataFrame
            reV supply curve results table including SC gid, lat/lon,
            res_gids, gid_counts.
        reeds_build : pd.DataFrame
            ReEDS buildout with rows for built capacity (MW) at each reV SC
            point. This should have columns: reeds_year, built_capacity, and
            sc_gid (corresponding to the reV supply curve point gid). Some
            cleaning of the column names will be performed for legacy tables
            but these are the column headers that are desired. This input can
            also include "plexos_node_gid" which will explicitly assign a
            supply curve point buildout to a single plexos node. If included,
            all points must be assigned to plexos nodes.
        atol : float
            Maximum difference in coord matching.

        Returns
        -------
        rev_sc : pd.DataFrame
            Same as input.
        reeds_build : pd.DataFrame
            Same as input but without lat/lon columns if matched.
        """
        join_on = 'sc_gid'
        reeds_build = reeds_build.sort_values(join_on)
        reeds_sc_gids = reeds_build[join_on].values
        rev_mask = rev_sc[join_on].isin(reeds_sc_gids)
        if not rev_mask.any():
            msg = ("There are no overlapping sc_gids between the provided reV "
                   "supply curve table the ReEDS buildout!")
            logger.error(msg)
            raise RuntimeError(msg)

        rev_sc = rev_sc.sort_values(join_on)

        rev_coord_labels = get_coord_labels(rev_sc)
        reeds_coord_labels = get_coord_labels(reeds_build)

        if rev_coord_labels is not None and reeds_coord_labels is not None:
            reeds_coords = reeds_build[reeds_coord_labels].values
            rev_coords = rev_sc.loc[rev_mask, rev_coord_labels].values

            check = np.allclose(reeds_coords, rev_coords, atol=atol, rtol=0.0)
            if not check:
                emsg = ('reV SC and REEDS Buildout coordinates do not match.')
                logger.exception(emsg)
                raise ValueError(emsg)

            reeds_build = reeds_build.drop(labels=reeds_coord_labels, axis=1)

        return rev_sc, reeds_build

    @classmethod
    def _parse_rev_reeds(cls, rev_sc, reeds_build, build_year=2050):
        """Parse and combine reV SC and REEDS buildout tables into single table

        Parameters
        ----------
        rev_sc : str | pd.DataFrame
            reV supply curve results table including SC gid, lat/lon,
            res_gids, gid_counts. Or  path to reV supply curve table.
        reeds_build : str | pd.DataFrame
            ReEDS buildout with rows for built capacity (MW) at each reV SC
            point. This should have columns: reeds_year, built_capacity, and
            sc_gid (corresponding to the reV supply curve point gid). Some
            cleaning of the column names will be performed for legacy tables
            but these are the column headers that are desired. This input can
            also include "plexos_node_gid" which will explicitly assign a
            supply curve point buildout to a single plexos node. If included,
            all points must be assigned to plexos nodes.
        build_year : int, optional
            REEDS year of interest, by default 2050

        Returns
        -------
        table : pd.DataFrame
            rev_sc and reeds_build inner joined on supply curve gid. This is
            basically the rev supply curve table paired down to only sc points
            that were built by reeds and that now includes the built_capacity
            column for each sc point in MW.
        """
        rev_sc = DataCleaner.rename_cols(
            parse_table(rev_sc),
            name_map=DataCleaner.REV_NAME_MAP)
        reeds_build = DataCleaner.rename_cols(
            parse_table(reeds_build),
            name_map=DataCleaner.REEDS_NAME_MAP)

        year_mask = (reeds_build['reeds_year'] == build_year)
        if not any(year_mask):
            msg = 'Build year {} not found in reeds data!'.format(build_year)
            logger.error(msg)
            raise ValueError(msg)

        reeds_build = reeds_build[year_mask]

        join_on = 'sc_gid'
        if 'sc_gid' not in rev_sc or 'sc_gid' not in reeds_build:
            raise KeyError('GID must be in reV SC and REEDS Buildout tables!')

        rev_sc, reeds_build = cls._check_rev_reeds_coordinates(rev_sc,
                                                               reeds_build)

        check_isin = np.isin(reeds_build[join_on].values,
                             rev_sc[join_on].values)
        if not all(check_isin):
            missing_cap = reeds_build.loc[check_isin, 'built_capacity']
            missing_cap = missing_cap.values.sum()
            total_cap = reeds_build['built_capacity'].values.sum()
            wmsg = ('There are REEDS buildout GIDs that are not in the reV '
                    'supply curve table: {} out of {} total REEDS buildout '
                    'sites which is {:.2f} MW missing out of {:.2f} MW total.'
                    .format(np.sum(~check_isin), len(reeds_build),
                            missing_cap, total_cap))
            warn(wmsg)
            logger.warning(wmsg)

        table = pd.merge(rev_sc, reeds_build, how='inner', left_on=join_on,
                         right_on=join_on)

        return table

    def _check_gids(self):
        """Ensure that the SC buildout GIDs are available in the cf file.

        Returns
        -------
        bad_sc_points : list
            List of missing supply curve gids
            (in reeds but not in reV resource).
        """
        bad_sc_points = []
        missing = list(set(self.sc_res_gids) - set(self.available_res_gids))
        if any(missing):
            wmsg = ('The CF file is missing {} resource gids that were built '
                    'in the REEDS-reV SC build out: {}'
                    .format(len(missing), missing))
            warn(wmsg)
            logger.warning(wmsg)

            gid_col = self.sc_build['res_gids'].values
            if isinstance(gid_col[0], str):
                gid_col = [json.loads(s) for s in gid_col]
            else:
                gid_col = list(gid_col)

            for i, sc_gids in enumerate(gid_col):
                if any(m in sc_gids for m in missing):
                    bad_sc_points.append(self.sc_build.iloc[i]['sc_gid'])

            wmsg = ('There are {} SC points with missing gids: {}'
                    .format(len(bad_sc_points), bad_sc_points))
            warn(wmsg)
            logger.warning(wmsg)

        return bad_sc_points

    def _handle_missing_resource_gids(self, bad_sc_points):
        """Merge requested capacity in missing SC gids into nearest good pixels

        Parameters
        ----------
        bad_sc_points : list
            List of missing supply curve gids
            (in reeds but not in reV resource).
        """
        if any(bad_sc_points):
            bad_bool = self.sc_build['sc_gid'].isin(bad_sc_points)
            bad_cap_arr = self.sc_build.loc[bad_bool, 'built_capacity'].values
            good_bool = ~bad_bool
            bad_cap = bad_cap_arr.sum()
            wmsg = ('{} MW of capacity is being merged from bad SC points.'
                    .format(bad_cap))
            warn(wmsg)
            logger.warning(wmsg)

            clabels = get_coord_labels(self.sc_build)
            # pylint: disable=not-callable
            good_tree = cKDTree(self.sc_build.loc[good_bool, clabels])
            _, i = good_tree.query(self.sc_build.loc[bad_bool, clabels])

            ilen = len(self.sc_build)
            icap = self.sc_build['built_capacity'].sum()

            add_index = self.sc_build.index.values[good_bool][i]

            for i, ai in enumerate(add_index):
                self.sc_build.loc[ai, 'built_capacity'] += bad_cap_arr[i]

            bad_ind = self.sc_build.index.values[bad_bool]
            self._sc_build = self._sc_build.drop(bad_ind, axis=0)

            olen = len(self.sc_build)
            ocap = self.sc_build['built_capacity'].sum()

            wmsg = ('SC build table reduced from {} to {} rows, '
                    'capacity from {} to {} (should be the same).'
                    .format(ilen, olen, icap, ocap))
            warn(wmsg)
            logger.warning(wmsg)

            cap_error = (icap - ocap) / icap
            if cap_error > 0.001:
                msg = ('Too much capacity is being lost due to missing '
                       'resource gids! Capacity difference is {}%. '
                       'Cannot continue.'.format(cap_error * 100))
                logger.error(msg)
                raise RuntimeError(msg)

    def _make_node_map(self):
        """Run ckdtree to map built rev SC points to plexos nodes.

        Returns
        -------
        plx_node_index : np.ndarray
            KDTree query output, (n, 1) array of plexos node indices mapped to
            the SC builds where n is the number of SC points built.
            Each value in this array gives the plexos node index that the sc
            point is mapped to. So self.node_map[10] yields the plexos node
            index for self.sc_build[10].
        """

        if isinstance(self._plexos_nodes, GeoDataFrame):
            logger.info('Found plexos node shape files, assigning nodes '
                        'based on shapes containing reV supply curve points.')
            temp = RegionClassifier.run(self.sc_build, self._plexos_nodes,
                                        regions_label='plexos_id',
                                        force=self._force_shape_map)
            plx_node_index = temp['plexos_id'].values.astype(int)
            if any(plx_node_index < 0):
                msg = ('Could not find a matching shape for {} supply curve '
                       'points: \n{}'
                       .format((plx_node_index < 0).sum(),
                               self.sc_build[(plx_node_index < 0)]))
                logger.error(msg)
                raise RuntimeError(msg)

        elif 'plexos_node_gid' in self.sc_build:
            if 'gid' not in self._plexos_nodes:
                msg = ('"plexos_node_gid" was found in the reV/ReEDS supply '
                       'curve buildout tables for explicit node assignment '
                       'but "gid" was not found in the plexos node table.')
                logger.error(msg)
                raise KeyError(msg)

            logger.info('Found "plexos_node_gid" in the reV/ReEDS buildout '
                        'tables and "gid" in the plexos node tables, '
                        'performing explicitly defined node assignment.')
            assigned_nodes = set(self.sc_build['plexos_node_gid']
                                 .values.astype(str))

            missing = [n for n in assigned_nodes
                       if n not in
                       self._plexos_nodes['gid'].values.astype(str)]
            if any(missing):
                msg = ('reV/ReEDS assigned supply curve buildouts to the '
                       'following nodes that were not found in the plexos '
                       'node table: {}'.format(missing))
                print(self._plexos_nodes['gid'].astype(str))
                logger.error(msg)
                raise ValueError(msg)

            na_mask = pd.isna(self.sc_build['plexos_node_gid'])
            if any(na_mask):
                msg = ('Some supply curve buildouts were not assigned a value '
                       'in the "plexos_node_gid" column. If explicitly '
                       'assigning sc points to plexos nodes, all sc points '
                       'must be assigned: {}'.format(self.sc_build[na_mask]))
                logger.error(msg)
                raise ValueError(msg)

            plx_tmp = self._plexos_nodes[['gid']].astype(str)
            plx_tmp['plx_node_index'] = np.arange(len(plx_tmp))
            sc_tmp = self.sc_build[['plexos_node_gid']].astype(str)
            join_tmp = pd.merge(sc_tmp, plx_tmp, how='left',
                                left_on='plexos_node_gid', right_on='gid')
            plx_node_index = join_tmp['plx_node_index'].values

        else:
            logger.info('Assigning built reV supply curve points to plexos '
                        'nodes based on KDTree nearest neighbor distance.')
            plexos_coord_labels = get_coord_labels(self._plexos_nodes)
            sc_coord_labels = get_coord_labels(self.sc_build)
            # pylint: disable=not-callable
            tree = cKDTree(self._plexos_nodes[plexos_coord_labels])
            d, plx_node_index = tree.query(self.sc_build[sc_coord_labels], k=1)
            logger.info('Plexos Node KDTree distance min / mean / max: '
                        '{} / {} / {}'
                        .format(np.round(d.min(), decimals=3),
                                np.round(d.mean(), decimals=3),
                                np.round(d.max(), decimals=3)))

        if len(plx_node_index.shape) == 1:
            plx_node_index = plx_node_index.reshape((len(plx_node_index), 1))

        return plx_node_index

    def make_profiles(self):
        """Make a 2D array of aggregated plexos gen profiles.

        Returns
        -------
        profiles : np.ndarray
            (t, n) array of Plexos node generation profiles where t is the
            timeseries length and n is the number of plexos nodes.
        """

        if self.max_workers != 1:
            profiles = self._make_profiles_parallel()
        else:
            profiles = self._make_profiles_serial()

        return profiles

    def _make_profiles_parallel(self):
        """Make a 2D array of aggregated plexos gen profiles in parallel.

        Returns
        -------
        profiles : np.ndarray
            (t, n) array of Plexos node generation profiles where t is the
            timeseries length and n is the number of plexos nodes.
        """
        profiles = self._init_output(self.n_plexos_nodes)
        progress = 0
        futures = {}
        loggers = [__name__, 'reVX']
        with SpawnProcessPool(max_workers=self.max_workers,
                              loggers=loggers) as exe:
            for i, inode in enumerate(np.unique(self.node_map)):
                mask = (self.node_map == inode)
                f = exe.submit(PlexosNode.run,
                               self.sc_build[mask], self._cf_fpath,
                               res_gids=self.available_res_gids,
                               forecast_fpath=self._forecast_fpath,
                               forecast_map=self._forecast_map,
                               force_full_build=self._force_full_build)
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
                    logger.info('{} % of plexos node profiles built.'
                                .format(progress))

        return profiles

    def _make_profiles_serial(self):
        """Make a 2D array of aggregated plexos gen profiles in serial.

        Returns
        -------
        profiles : np.ndarray
            (t, n) array of Plexos node generation profiles where t is the
            timeseries length and n is the number of plexos nodes.
        """
        profiles = self._init_output(self.n_plexos_nodes)
        progress = 0
        for i, inode in enumerate(np.unique(self.node_map)):
            mask = (self.node_map == inode)
            p = PlexosNode.run(
                self.sc_build[mask], self._cf_fpath,
                res_gids=self.available_res_gids,
                forecast_fpath=self._forecast_fpath,
                forecast_map=self._forecast_map,
                force_full_build=self._force_full_build)

            profile, sc_gids, res_gids, gen_gids, res_built = p
            profiles[:, i] = profile
            self._ammend_output_meta(i, sc_gids, res_gids, gen_gids, res_built)

            current_prog = ((i + 1)
                            // (len(np.unique(self.node_map)) / 100))
            if current_prog > progress:
                progress = current_prog
                logger.info('{} % of plexos node profiles built.'
                            .format(progress))

        return profiles

    @classmethod
    def run(cls, plexos_nodes, rev_sc, reeds_build, cf_fpath,
            forecast_fpath=None, build_year=2050, plexos_columns=None,
            force_full_build=False, force_shape_map=False,
            plant_name_col=None, tech_tag=None, timezone='UTC',
            out_fpath=None, max_workers=None):
        """Run plexos aggregation.

        Parameters
        ----------
        plexos_nodes : str | pd.DataFrame
            Plexos node meta data including gid, latitude, longitude, voltage.
            Or file path to .csv containing plexos node meta data, or a file
            path to a .shp file that contains plexos nodes defined as shapes.
        rev_sc : str | pd.DataFrame
            reV supply curve results table including SC gid, latitude,
            longitude, res_gids, gid_counts. Or file path to reV supply
            curve table. Note that the gen_gids column in the rev_sc is ignored
            and only the res_gids from rev_sc are mapped to the corresponding
            "gid" column in the cf_fpath meta data.
        reeds_build : pd.DataFrame
            ReEDS buildout with rows for built capacity (MW) at each reV SC
            point. This should have columns: reeds_year, built_capacity, and
            sc_gid (corresponding to the reV supply curve point gid). Some
            cleaning of the column names will be performed for legacy tables
            but these are the column headers that are desired. This input can
            also include "plexos_node_gid" which will explicitly assign a
            supply curve point buildout to a single plexos node. If included,
            all points must be assigned to plexos nodes.
        cf_fpath : str
            File path to capacity factor file (reV gen output) to
            get profiles from.
        forecast_fpath : str | None
            Forecasted capacity factor .h5 file path (reV results). If not
            None, the supply curve res_gids are mapped to sites in the
            cf_fpath, then the coordinates from cf_fpath are mapped to the
            nearest neighbor sites in the forecast_fpath, where the final
            generation profiles are retrieved from.
        build_year : int
            REEDS year of interest.
        plexos_columns : list | None
            Additional columns from the plexos_nodes input to pass through
            to the output meta data.
        force_full_build : bool
            Flag to ensure the full requested buildout is built at each SC
            point. If True, the remainder of the requested build will always
            be built at the last resource gid in the sc point.
        force_shape_map : bool
            Flag to force the mapping of supply curve points to the plexos
            node shape file input (if a shape file is input) via nearest
            neighbor to shape centroid.
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
        out_fpath : str, optional
            Path to .h5 file into which plant buildout should be saved. A
            plexos-formatted csv will also be written in the same directory.
            By default None.
        max_workers : int | None
            Max workers for parallel profile aggregation. None uses all
            available workers. 1 will run in serial.

        Returns
        -------
        plexos_meta : pd.DataFrame
            Plexos node meta data with built capacities.
        time_index : pd.datetimeindex
            Time index for the profiles.
        profiles : np.ndarray
            Generation profile timeseries at each plexos node.
        """

        pa = cls(plexos_nodes, rev_sc, reeds_build, cf_fpath,
                 forecast_fpath=forecast_fpath,
                 build_year=build_year,
                 plexos_columns=plexos_columns,
                 force_full_build=force_full_build,
                 force_shape_map=force_shape_map,
                 plant_name_col=plant_name_col,
                 tech_tag=tech_tag,
                 timezone=timezone,
                 max_workers=max_workers)

        profiles = pa.make_profiles()

        if out_fpath is not None:
            pa.export(pa.plexos_meta, pa.time_index, profiles, out_fpath)

        return pa.plexos_meta, pa.time_index, profiles


class RevReedsPlexosManager:
    """rev-reeds-plexos job manager."""

    def __init__(self, plexos_nodes, rev_sc, reeds_build, cf_fpath,
                 forecast_fpath=None, wait=300, db_host='gds_edit.nrel.gov',
                 db_user=None, db_pass=None, db_port=5432):
        """
        Parameters
        ----------
        plexos_nodes : str | pd.DataFrame
            Plexos node meta data (CSV/SHP file path or database.schema.name)
        rev_sc : str | pd.DataFrame
            reV supply curve results (CSV file path or database.schema.name)
        reeds_build : str | pd.DataFrame
            REEDS buildout results (CSV file path or database.schema.name)
            ReEDS buildout with rows for built capacity (MW) at each reV SC
            point. This should have columns: reeds_year, built_capacity, and
            sc_gid (corresponding to the reV supply curve point gid). Some
            cleaning of the column names will be performed for legacy tables
            but these are the column headers that are desired. This input can
            also include "plexos_node_gid" which will explicitly assign a
            supply curve point buildout to a single plexos node. If included,
            all points must be assigned to plexos nodes.
        cf_fpath : str
            File path to capacity factor file (reV gen output) to
            get profiles from.
        forecast_fpath : str | None
            Forecasted capacity factor .h5 file path (reV results). If not
            None, the supply curve res_gids are mapped to sites in the
            cf_fpath, then the coordinates from cf_fpath are mapped to the
            nearest neighbor sites in the forecast_fpath, where the final
            generation profiles are retrieved from.
        wait : int
            Integer seconds to wait for DB connection to become available
            before raising exception.
        db_host : str
            Database host name.
        db_user : str
            Your database user name.
        db_pass : str
            Database password (None if your password is cached).
        db_port : int
            Database port.
        """
        self.plexos_nodes = parse_table_name(plexos_nodes, wait=wait,
                                             db_host=db_host,
                                             db_user=db_user,
                                             db_pass=db_pass,
                                             db_port=db_port)
        self.plexos_nodes = DataCleaner.pre_filter_plexos_meta(
            self.plexos_nodes)

        self.rev_sc = parse_table_name(rev_sc, wait=wait,
                                       db_host=db_host,
                                       db_user=db_user,
                                       db_pass=db_pass,
                                       db_port=db_port)
        self.reeds_build = parse_table_name(reeds_build, wait=wait,
                                            db_host=db_host,
                                            db_user=db_user,
                                            db_pass=db_pass,
                                            db_port=db_port)

        self.rev_sc = DataCleaner.rename_cols(
            self.rev_sc, name_map=DataCleaner.REV_NAME_MAP)
        self.reeds_build = DataCleaner.rename_cols(
            self.reeds_build, name_map=DataCleaner.REEDS_NAME_MAP)

        self.cf_fpath = cf_fpath
        if not os.path.exists(self.cf_fpath):
            raise FileNotFoundError('Could not find cf_fpath: {}'
                                    .format(cf_fpath))

        self.forecast_fpath = forecast_fpath
        if self.forecast_fpath is not None:
            if not os.path.exists(self.forecast_fpath):
                raise FileNotFoundError('Could not find forecast_fpath: {}'
                                        .format(forecast_fpath))

    @classmethod
    def main(cls, plexos_nodes, rev_sc, reeds_build, cf_fpath,
             forecast_fpath=None, agg_kwargs=None, wait=300,
             db_host='gds_edit.nrel.gov', db_user=None, db_pass=None,
             db_port=5432):
        """Run the Plexos pipeline for a single extent.

        Parameters
        ----------
        plexos_nodes : str | pd.DataFrame
            Plexos node meta data (CSV/SHP file path or database.schema.name)
        rev_sc : str | pd.DataFrame
            reV supply curve results (CSV file path or database.schema.name)
        reeds_build : str | pd.DataFrame
            REEDS buildout results (CSV file path or database.schema.name)
            ReEDS buildout with rows for built capacity (MW) at each reV SC
            point. This should have columns: reeds_year, built_capacity, and
            sc_gid (corresponding to the reV supply curve point gid). Some
            cleaning of the column names will be performed for legacy tables
            but these are the column headers that are desired. This input can
            also include "plexos_node_gid" which will explicitly assign a
            supply curve point buildout to a single plexos node. If included,
            all points must be assigned to plexos nodes.
        cf_fpath : str
            File path to capacity factor file (reV gen output) to
            get profiles from.
        forecast_fpath : str | None
            Forecasted capacity factor .h5 file path (reV results). If not
            None, the supply curve res_gids are mapped to sites in the
            cf_fpath, then the coordinates from cf_fpath are mapped to the
            nearest neighbor sites in the forecast_fpath, where the final
            generation profiles are retrieved from.
        agg_kwargs : dict
            Optional additional kwargs for the aggregation run.
        wait : int
            Integer seconds to wait for DB connection to become available
            before raising exception.
        db_host : str
            Database host name.
        db_user : str
            Your database user name.
        db_pass : str
            Database password (None if your password is cached).
        db_port : int
            Database port.

        Returns
        -------
        meta : pd.DataFrame
            Plexos node meta data.
        time_index : pd.Datetimeindex
            Time index.
        profiles : np.ndarray
            Plexos node generation profiles.
        """

        meta = None
        time_index = None
        profiles = None

        if agg_kwargs is None:
            agg_kwargs = {}

        logger.info('Running PLEXOS aggregation with plexos nodes input: {}'
                    .format(plexos_nodes))
        logger.info('Running PLEXOS aggregation with reV SC input: {}'
                    .format(rev_sc))
        logger.info('Running PLEXOS aggregation with REEDS input: {}'
                    .format(reeds_build))
        logger.info('Running PLEXOS aggregation with reV Gen input: {}'
                    .format(cf_fpath))
        logger.info('Running PLEXOS aggregation with forecast filepath: {}'
                    .format(forecast_fpath))

        pm = cls(plexos_nodes, rev_sc, reeds_build, cf_fpath,
                 forecast_fpath=forecast_fpath, wait=wait,
                 db_host=db_host, db_user=db_user, db_pass=db_pass,
                 db_port=db_port)

        try:
            meta, time_index, profiles = PlexosAggregation.run(
                pm.plexos_nodes, pm.rev_sc, pm.reeds_build, pm.cf_fpath,
                forecast_fpath=pm.forecast_fpath, **agg_kwargs)

        except Exception as e:
            logger.exception(e)
            raise e

        return meta, time_index, profiles

    @classmethod
    def _run_group(cls, df_group, cf_year, build_year, plexos_columns=None,
                   force_full_build=False, force_shape_map=False):
        """Run a group of plexos node aggregations all belonging to the same
        final extent.

        Parameters
        ----------
        df_group : str
            DataFrame from the job_file with a common group.
        cf_year : str
            Year of the cf_fpath resource year (will be inserted if {} is in
            cf_fpath).
        build_years : list | tuple
            REEDS years to run scenarios for.
        plexos_columns : list | None
            Additional columns from the plexos_nodes input to pass through
            to the output meta data.
        force_full_build : bool
            Flag to ensure the full requested buildout is built at each SC
            point. If True, the remainder of the requested build will always
            be built at the last resource gid in the sc point.
        force_shape_map : bool
            Flag to force the mapping of supply curve points to the plexos
            node shape file input (if a shape file is input) via nearest
            neighbor to shape centroid.

        Returns
        -------
        meta : pd.DataFrame
            Plexos node meta data.
        time_index : pd.Datetimeindex
            Time index.
        profiles : np.ndarray
            Plexos node generation profiles.
        """

        dc = None

        for i in df_group.index.values:
            plexos_nodes = df_group.loc[i, 'plexos_nodes']
            reeds_build = df_group.loc[i, 'reeds_build']
            cf_fpath = df_group.loc[i, 'cf_fpath']
            if '{}' in cf_fpath:
                cf_fpath = cf_fpath.format(cf_year)
            elif cf_year not in cf_fpath:
                warn('Specified CF year {} not present in cf file string: {}'
                     .format(cf_year, cf_fpath))

            rev_sc = df_group.loc[i, 'rev_sc']

            forecast_fpath = None
            if 'forecast_fpath' in df_group:
                forecast_fpath = df_group.loc[i, 'forecast_fpath']
                if '{}' in forecast_fpath:
                    forecast_fpath = forecast_fpath.format(cf_year)
                elif cf_year not in forecast_fpath:
                    warn('Specified CF year {} not present in ECMWF file '
                         'string: {}'.format(cf_year, forecast_fpath))

            agg_kwargs = {'build_year': build_year,
                          'plexos_columns': plexos_columns,
                          'force_full_build': force_full_build,
                          'force_shape_map': force_shape_map}
            meta, ti, profiles = cls.main(plexos_nodes, rev_sc, reeds_build,
                                          cf_fpath, agg_kwargs=agg_kwargs,
                                          forecast_fpath=forecast_fpath)

            if meta is None:
                e = ('Plexos aggregation manager failed. '
                     'PlexosAggregation.run() '
                     'failed to create a meta data object.')
                logger.error(e)
                raise RuntimeError(e)
            else:
                if dc is None:
                    dc = DataCleaner(meta, profiles)
                else:
                    dc.merge_extent(meta, profiles)

        meta, profiles = dc.merge_small()

        return meta, ti, profiles

    @classmethod
    def run(cls, job, out_dir, scenario=None, cf_year=2012,
            build_years=(2024, 2050), plexos_columns=None,
            force_full_build=False, force_shape_map=False):
        """Run plexos node aggregation for a job file input.

        Parameters
        ----------
        job : str | pd.DataFrame
            CSV file with plexos aggregation job config. Needs the following
            columns: (scenario, group, cf_fpath, reeds_build, rev_sc,
            plexos_nodes)
        out_dir : str
            Path to an output directory.
        scenario : str | None
            Optional filter to run plexos aggregation for just one scenario in
            the job.
        cf_year : str
            Year of the cf_fpath resource year (will be inserted if {} is in
            cf_fpath).
        build_years : list | tuple | int
            REEDS years to run scenarios for.
        plexos_columns : list | None
            Additional columns from the plexos_nodes input to pass through
            to the output meta data.
        force_full_build : bool
            Flag to ensure the full requested buildout is built at each SC
            point. If True, the remainder of the requested build will always
            be built at the last resource gid in the sc point.
        force_shape_map : bool
            Flag to force the mapping of supply curve points to the plexos
            node shape file input (if a shape file is input) via nearest
            neighbor to shape centroid.
        """

        if isinstance(job, str):
            job = pd.read_csv(job)

        job = job.where(pd.notnull(job), None)

        if isinstance(build_years, int):
            build_years = [build_years]

        if scenario is not None:
            job = job[(job['scenario'] == scenario)]

        for scenario, df_scenario in job.groupby('scenario'):
            logger.info('Running scenario "{}"'.format(scenario))
            for build_year in build_years:
                logger.info('Running build year {}'.format(build_year))
                fn_out = '{}_{}_{}.h5'.format(scenario, build_year,
                                              cf_year)
                out_fpath = os.path.join(out_dir, fn_out)

                if os.path.exists(out_fpath):
                    logger.info('Skipping exists: {}'.format(out_fpath))
                else:

                    for group, df_group in df_scenario.groupby('group'):
                        logger.info('Running group "{}"'.format(group))

                        meta, time_index, profiles = cls._run_group(
                            df_group, cf_year, build_year,
                            plexos_columns=plexos_columns,
                            force_full_build=force_full_build,
                            force_shape_map=force_shape_map)

                        logger.info('Saving result for group "{}" to file: {}'
                                    .format(group, out_fpath))

                        with Outputs(out_fpath, mode='a') as out:
                            meta = to_records_array(meta)
                            time_index = time_index.astype(str)
                            dtype = "S{}".format(len(time_index[0]))
                            time_index = np.array(time_index, dtype=dtype)
                            out._create_dset('{}/meta'.format(group),
                                             meta.shape,
                                             meta.dtype,
                                             data=meta)
                            out._create_dset('{}/time_index'.format(group),
                                             time_index.shape,
                                             time_index.dtype,
                                             data=time_index)
                            out._create_dset('{}/gen_profiles'.format(group),
                                             profiles.shape,
                                             profiles.dtype,
                                             chunks=(None, 100),
                                             data=profiles)

        logger.info('Plexos aggregation complete!')
