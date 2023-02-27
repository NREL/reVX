# -*- coding: utf-8 -*-
"""
Created on Tues Feb 2023

@author: bbenton
"""
import logging
import pandas as pd
import numpy as np
import os
import shutil
from warnings import warn
import re

from reVX.plexos.rev_reeds_plexos import PlexosAggregation
from rex import Outputs, Resource


logger = logging.getLogger(__name__)


class PrasAggregation(PlexosAggregation):
    """
    Framework to aggregate reV gen profiles to PRAS node power profiles and
    overwrite exisiting pras file with new generation profiles. This class
    takes as input the plexos nodes meta data (lat/lon or shape files), rev
    supply curve table, and reeds buildout table (specifying which rev sc
    points were built and at what capacity). The class will build power
    profiles for each supply curve point and then aggregate the sc point
    profiles to the nearest neighbor pras node (if pras nodes are defined by
    lat/lon) or the shape intersect pras node (if pras nodes are defined by
    shape file). The corresponding node power profiles in the pras output file
    will be overwritten with these new aggregated generation profiles.

    Examples
    --------

    >>> from reVX.pras.rev_reeds_pras import PrasAggregation
    >>>
    >>> node_shape_file = ...
    >>> rev_supply_curve_file = ...
    >>> reeds_build_file = ...
    >>> capacity_factor_file = ...
    >>> pras_file = ...
    >>>
    >>> PrasAggregation.run(node_shape_file, rev_supply_curve_file,
                            reeds_build_file, capacity_factor_file,
                            pras_file, build_year=2047, tech_type='upv')

    This will replace all the pras generation profiles with upv tech in the
    file pras_file.pras.h5 with aggregated generation profiles from cf_path.h5

    >>> reeds_build_files = [...]
    >>> capacity_factor_files = [...]
    >>> tech_types = ['upv', 'dupv', 'wind-ons', 'wind-ofs']
    >>>
    >>> PrasAggregation.run(node_shape_file, rev_supply_curve_file,
                            reeds_build_files, capacity_factor_files,
                            pras_file, build_year=2047,
                            tech_type=tech_types)

    This will run the same routine but for each tech type in tech_types. The
    files in reeds_build_files and capacity_factor_files have to correspond to
    the tech type in the tech_types list.
    """
    def __init__(self, plexos_nodes, rev_sc, reeds_build, cf_fpath, pras_file,
                 load_path=None, forecast_fpath=None, build_year=2050,
                 plexos_columns=None, force_full_build=False,
                 force_shape_map=False, plant_name_col=None, tech_tag=None,
                 res_class=None, tech_type=None, timezone='UTC',
                 dset_tag=None, max_workers=None):
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
        pras_file : str
            File path to existing PRAS file which will be amended to include
            generation data from given resource data.
        forecast_fpath : str | None
            Forecasted capacity factor .h5 file path (reV results). If not
            None, the supply curve res_gids are mapped to sites in the
            cf_fpath, then the coordinates from cf_fpath are mapped to the
            nearest neighbor sites in the forecast_fpath, where the final
            generation profiles are retrieved from.
        load_path : str | None
            Optional path to load data used to overwrite the 'regions/load'
            dataset in the pras_file.
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
        res_class : int | None
            Optional resource class to use to filter supply curve points.
            For example, if res_class = 3 then only supply curve points with
            class 3 will be kept in the sc_build table.
        tech_type : str | None
            Pras files will have the tech type in the
            ['generation/_core']['name'] entry, which is used to select the
            correct output index
        timezone : str
            Timezone for output generation profiles. This is a string that will
            be passed to pytz.timezone() e.g. US/Pacific, US/Mountain,
            US/Central, US/Eastern, or UTC. For a list of all available
            timezones, see pytz.all_timezones
        dset_tag : str
            Dataset tag to append to dataset names in cf profile file. e.g. If
            the cf profile file is a multi year file using dset_tag="-2008"
            will enable us to select the corresponding datasets
            (cf_mean-2008, cf_profile-2008, etc)
        max_workers : int | None
            Max workers for parallel profile aggregation. None uses all
            available workers. 1 will run in serial.
        """
        self._dset_tag = dset_tag if dset_tag is not None else ""
        self._tech_type = tech_type
        self._region_type = 'rb' if 'upv' in tech_type else 'rs'
        self._pras_file = pras_file
        self._pras_meta = None
        self._load_path = load_path
        super().__init__(plexos_nodes, rev_sc, reeds_build, cf_fpath,
                         forecast_fpath=forecast_fpath,
                         build_year=build_year,
                         plexos_columns=plexos_columns,
                         force_full_build=force_full_build,
                         force_shape_map=force_shape_map,
                         plant_name_col=plant_name_col,
                         tech_tag=tech_tag, res_class=res_class,
                         timezone=timezone, dset_tag=self._dset_tag,
                         max_workers=max_workers)
        logger.info('Running aggregation for tech_type={}, res_class={}, '
                    'rev_sc={}, reeds_build={}, cf_fpath={}, dset_tag={}.'
                    .format(self._tech_type, self._res_class, rev_sc,
                            reeds_build, cf_fpath, dset_tag))
        logger.info('Found {} pras_nodes and {} sc_build_nodes'
                    .format(len(self.pras_nodes), len(self.sc_build_nodes)))
        self.missing_nodes = self.get_missing_nodes()
        self.found_nodes = list(set(self.pras_nodes) - set(self.missing_nodes))

    @classmethod
    def get_pras_meta(cls, pras_file):
        """
        Get pras 'meta' which describes the tech type and class at each node

        Parameters
        ----------
        pras_file : str
            Path to the pras file being overwritten with new generation
            profiles

        Returns
        -------
        pd.DataFrame
        """
        with Outputs(pras_file, mode='r') as out:
            pras_meta = pd.DataFrame(out['generators/_core'])
            for col in pras_meta.columns:
                pras_meta[col] = pras_meta[col].apply(
                    lambda x: x.decode('utf-8'))
        return pras_meta

    @classmethod
    def get_pras_classes(cls, pras_file, tech_type):
        """
        Get list of tech classes present in the pras file for the requested
        technology type

        Parameters
        ----------
        pras_file : str
            Path to the pras file being overwritten with new generation
            profiles
        tech_type : str
            Technology type for which profiles are being overwritten

        Returns
        -------
        List
        """
        pras_meta = cls.get_pras_meta(pras_file)
        class_list = [int(x.split('_')[-1]) for x in pras_meta['category']
                      if x.startswith(tech_type)]
        class_list = np.unique(class_list)
        logger.info('Found classes {} for tech_type {}'
                    .format(class_list, tech_type))
        return class_list

    @property
    def region_node_map(self):
        """
        Get the map between node index and region label. i.e. node_index=1,
        region_label=p10
        """
        region_node_map = dict(
            zip(self._plexos_nodes[self._region_type].values,
                self._plexos_nodes['plexos_id'].values))
        return region_node_map

    @property
    def sc_build_nodes(self):
        """
        Get the labels for the nodes in the node_map. For example, the node_map
        might have an index of 1 for the region but this corresponds to the
        label of p10.
        """
        indices = np.unique(self.node_map)
        build_nodes = self._plexos_nodes[self._region_type].iloc[indices]
        build_nodes = build_nodes.values
        return build_nodes

    @property
    def pras_meta(self):
        """
        Get pras 'meta' which describes the tech type and class at each node

        Returns
        -------
        pd.DataFrame
        """
        if self._pras_meta is None:
            self._pras_meta = self.get_pras_meta(self._pras_file)
            self._pras_meta = self.enforce_tech_constraint()
        return self._pras_meta

    @property
    def pras_nodes(self):
        """
        Get list of available nodes in pras output file. For upv and dupv
        regions are labeled p1, p2, etc. For wind and csp regions are labeled
        s1, s2, etc.

        Returns
        -------
        List
        """
        def node_filter(x):
            out = re.search(r'p\d+', x)
            if out is None:
                out = re.search(r's\d+', x)
            return out[0]

        pras_nodes = self.pras_meta['name'].apply(node_filter).values
        return pras_nodes

    @property
    def pras_indices(self):
        """
        Get the list of pras file indices for the pras nodes available in the
        supply curve node list.

        Returns
        -------
        np.ndarray
        """
        node_mask = np.isin(self.pras_nodes, self.sc_build_nodes)
        pras_indices = list(self.pras_meta.index[node_mask])
        return pras_indices

    @property
    def sc_build_indices(self):
        """
        Get the list of profile indices corresponding to the sc_build_nodes -
        used to write the new profiles to the pras output.

        Returns
        -------
        List
        """
        sc_build_indices = [i for i, x in enumerate(self.sc_build_nodes)
                            if x in self.pras_nodes]
        return sc_build_indices

    def enforce_tech_constraint(self):
        """
        Filter the pras generator meta for the requested technology type and
        class

        Returns
        -------
        pd.DataFrame
        """
        tech_key = '{}_'.format(self._tech_type)
        if self._res_class is not None:
            tech_key += '{}'.format(self._res_class)
        category_mask = [x.startswith(tech_key)
                         for x in self._pras_meta['category']]
        self._pras_meta = self._pras_meta[category_mask]
        if self._pras_meta.empty:
            msg = ('Found zero pras nodes with requested tech_key {}.'
                   .format(tech_key))
            logger.error(msg)
            raise RuntimeError(msg)
        return self._pras_meta

    def get_missing_nodes(self):
        """
        Check for nodes present in the pras output file which cannot be found
        in the supply curve node list
        """
        node_mask = np.isin(self.pras_nodes, self.sc_build_nodes)
        missing_nodes = sorted([x for x in np.unique(self.pras_nodes)
                                if x not in self.sc_build_nodes])
        msg = ('{} / {} pras nodes not found in sc_build_nodes, for '
               'tech_key {}_{}. Missing nodes: {}.').format(
                   len(missing_nodes), len(self.pras_nodes), self._tech_type,
                   self._res_class, missing_nodes)
        if not all(node_mask):
            logger.warning(msg)
            warn(msg)
        else:
            logger.info('All {} pras nodes found in sc_build_nodes'
                        .format(len(self.pras_nodes)))
        return missing_nodes

    def export(self, meta, time_index, profiles, out_fpath):
        """Export generation profiles to h5 and plexos-formatted csv

        Parameters
        ----------
        plant_meta : pd.DataFrame
            Plant / pras node meta data with built capacities and mappings to
            the resource used.
        time_index : pd.datetimeindex
            Time index for the profiles.
        profiles : np.ndarray
            Generation profile timeseries in MW at each plant / pras node.
        out_fpath : str
            Path to .h5 file into which plant buildout should be saved. A
            pras-formatted csv will also be written in the same directory.
        """
        if not out_fpath.endswith('.h5'):
            out_fpath = out_fpath + '.h5'

        msg = ("out_fpath={} does not exist. This file must come from a "
               "previous ReEDS2PRAS run").format(out_fpath)
        assert os.path.exists(out_fpath), msg

        new_out_file = out_fpath.replace('.h5', '_updated.h5')
        if not os.path.exists(new_out_file):
            shutil.copy(out_fpath, new_out_file)
        out_fpath = new_out_file

        logger.info('Saving profiles for {}_{} to file: {}'
                    .format(self._tech_type, self._res_class, out_fpath))

        profiles = self.tz_convert_profiles(profiles, self._timezone)

        with Outputs(out_fpath, mode='a') as out:
            out['generators/capacity'][:, self.pras_indices] = \
                profiles[:, self.sc_build_indices]

        names = np.arange(profiles.shape[1])
        if self._plant_name_col is not None:
            names = self.get_unique_plant_names(meta, self._plant_name_col,
                                                self._tech_tag)

        df_plx = pd.DataFrame(profiles, columns=names,
                              index=time_index.tz_convert(None))
        df_plx.index.name = 'DATETIME'
        csv_fp = out_fpath.replace('.h5', '_{}_{}.csv'.format(self._tech_type,
                                                              self._res_class))
        df_plx.to_csv(csv_fp)

        logger.info('Wrote pras formatted profiles to: {}'.format(csv_fp))

    @classmethod
    def write_load_data(cls, load_path, out_fpath):
        """Export load data to pras file

        Parameters
        ----------
        load_path : str
            Path to load data to use for overwriting 'regions/load' dataset
            in out_fpath.
        out_fpath : str
            Path to .h5 file into which plant buildout should be saved. A
            pras-formatted csv will also be written in the same directory.
        """
        if not out_fpath.endswith('.h5'):
            out_fpath = out_fpath + '.h5'

        msg = ("out_fpath={} does not exist. This file must come from a "
               "previous ReEDS2PRAS run").format(out_fpath)
        assert os.path.exists(out_fpath), msg

        new_out_file = out_fpath.replace('.h5', '_updated.h5')
        if not os.path.exists(new_out_file):
            shutil.copy(out_fpath, new_out_file)
        out_fpath = new_out_file

        logger.info('Saving load from {} to {}'.format(load_path, out_fpath))

        with Outputs(out_fpath, mode='a') as out:
            with Resource(load_path) as res:
                out['regions/load'] = res['load']

    def _make_node_map(self):
        """Map built rev SC points to pras nodes.

        Returns
        -------
        pras_node_index : np.ndarray
            (n, 1) array of pras node indices mapped to the SC builds where n
            is the number of SC points built.  Each value in this array gives
            the pras node index that the sc point is mapped to. So
            self.node_map[10] yields the pras node index for
            self.sc_build[10].
        """
        if 'region' in self.sc_build:
            pras_node_index = self.sc_build['region'].values
            pras_node_index = [self.region_node_map[k] for k
                               in pras_node_index]
            pras_node_index = np.array(pras_node_index)

            if len(pras_node_index.shape) == 1:
                pras_node_index = pras_node_index.reshape(
                    (len(pras_node_index), 1))
        else:
            msg = '"region" column not found in sc_build.'
            logger.error(msg)
            raise RuntimeError(msg)
        return pras_node_index

    @classmethod
    def run(cls, plexos_nodes, rev_sc, reeds_build, cf_fpath, pras_file,
            load_path=None, forecast_fpath=None, build_year=2050,
            plexos_columns=None, force_full_build=False, force_shape_map=False,
            plant_name_col=None, tech_tag=None, res_class=None, tech_type=None,
            timezone='UTC', dset_tag=None, max_workers=None):
        """Run pras aggregation and output for the requested tech types. This
        will aggregate the generation profiles over region specific supply
        curve points found in rev_sc, after filtering rev_sc for the requested
        tech type, and resource class. These aggregated generation profiles
        will then be written to the pras output file in the location
        corresponding to the region, tech type, and class.

        plexos_nodes : str | pd.DataFrame
            Plexos node meta data including gid, latitude, longitude, voltage.
            Or file path to .csv containing plexos node meta data, or a file
            path to a .shp file that contains plexos nodes defined as shapes.
        rev_sc : str | pd.DataFrame
            reV supply curve results table including SC gid, latitude,
            longitude, res_gids, gid_counts. Or file path to reV supply
            curve table. If providing a list of tech types this also needs to
            be a list of the corresponding supply curve files.
        reeds_build : str | pd.DataFrame | list
            ReEDS buildout with rows for built capacity (MW) at each reV SC
            point. This should have columns: reeds_year, built_capacity, and
            sc_gid (corresponding to the reV supply curve point gid). Some
            cleaning of the column names will be performed for legacy tables
            but these are the column headers that are desired. This input can
            also include "plexos_node_gid" which will explicitly assign a
            supply curve point buildout to a single plexos node. If included,
            all points must be assigned to plexos nodes. If providing a list
            of tech types this also needs to be a list of the corresponding
            reeds builout files for each tech type.
        cf_fpath : str | list
            File path to capacity factor file (reV gen output) to
            get profiles from. If providing a list of tech types this also
            needs to be a list of the corresponding capacity factor files.
        pras_file : str
            File path to existing PRAS file which will be amended to include
            generation data from given resource data.
        load_path : str | None
            Optional path to load data used to overwrite the 'regions/load'
            dataset in the pras_file.
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
        res_class : int | None
            Optional resource class to use to filter supply curve points.
            For example, if res_class = 3 then only supply curve points with
            class 3 will be kept in the sc_build table. If None then the
            aggregation will be run for each class found in the pras output
            file
        tech_type : str | list
            Tech type for which to overwrite the generation profiles in the
            pras output file. Pras files will have the tech type in the
            ['generation/_core']['name'] entry, which is used to select the
            correct output index
        timezone : str
            Timezone for output generation profiles. This is a string that will
            be passed to pytz.timezone() e.g. US/Pacific, US/Mountain,
            US/Central, US/Eastern, or UTC. For a list of all available
            timezones, see pytz.all_timezones
        dset_tag : str
            Dataset tag to append to dataset names in cf profile file. e.g. If
            the cf profile file is a multi year file using dset_tag="-2008"
            will enable us to select the corresponding datasets
            (cf_mean-2008, cf_profile-2008, etc)
        max_workers : int | None
            Max workers for parallel profile aggregation. None uses all
            available workers. 1 will run in serial.
        """

        if not isinstance(reeds_build, list):
            reeds_build = [reeds_build]
        if not isinstance(cf_fpath, list):
            cf_fpath = [cf_fpath]
        if not isinstance(rev_sc, list):
            rev_sc = [rev_sc]
        if not isinstance(tech_type, list):
            tech_type = [tech_type]
        missing_nodes = []

        if load_path is not None:
            cls.write_load_data(load_path, pras_file)

        for i, tech in enumerate(tech_type):
            reeds_build_fp = reeds_build[i]
            cf_fp = cf_fpath[i]
            rev_sc_fp = rev_sc[i]
            cls._run_single(tech, plexos_nodes, rev_sc_fp, reeds_build_fp,
                            cf_fp, pras_file, forecast_fpath, build_year,
                            plexos_columns, force_full_build, force_shape_map,
                            plant_name_col, tech_tag, res_class, timezone,
                            dset_tag, max_workers, missing_nodes)

        logger.info('Missing {} nodes across all classes: {}'
                    .format(len(set(missing_nodes)), set(missing_nodes)))

    @classmethod
    def _run_single(cls, tech, plexos_nodes, rev_sc_fp, reeds_build_fp, cf_fp,
                    pras_file, forecast_fpath, build_year, plexos_columns,
                    force_full_build, force_shape_map, plant_name_col,
                    tech_tag, res_class, timezone, dset_tag, max_workers,
                    missing_nodes):
        """Run pras aggregation and output for a single requested tech type.
        This will aggregate the generation profiles over region specific supply
        curve points found in rev_sc, after filtering rev_sc for the requested
        tech type, and resource class. These aggregated generation profiles
        will then be written to the pras output file in the location
        corresponding to the region, tech type, and class.

        Parameters
        ----------
        tech : str
            Tech type for which to overwrite the generation profiles in the
            pras output file. Pras files will have the tech type in the
            ['generation/_core']['name'] entry, which is used to select the
            correct output index
        plexos_nodes : str | pd.DataFrame
            Plexos node meta data including gid, latitude, longitude, voltage.
            Or file path to .csv containing plexos node meta data, or a file
            path to a .shp file that contains plexos nodes defined as shapes.
        rev_sc_fp : str | pd.DataFrame
            reV supply curve results table including SC gid, latitude,
            longitude, res_gids, gid_counts. Or file path to reV supply
            curve table.
        reeds_build_fp : str | pd.DataFrame
            ReEDS buildout with rows for built capacity (MW) at each reV SC
            point. This should have columns: reeds_year, built_capacity, and
            sc_gid (corresponding to the reV supply curve point gid). Some
            cleaning of the column names will be performed for legacy tables
            but these are the column headers that are desired. This input can
            also include "plexos_node_gid" which will explicitly assign a
            supply curve point buildout to a single plexos node. If included,
            all points must be assigned to plexos nodes.
        cf_fp : str
            File path to capacity factor file (reV gen output) to
            get profiles from.
        pras_file : str
            File path to existing PRAS file which will be amended to include
            generation data from given resource data.
        load_path : str | None
            Optional path to load data used to overwrite the 'regions/load'
            dataset in the pras_file.
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
        res_class : int | None
            Optional resource class to use to filter supply curve points.
            For example, if res_class = 3 then only supply curve points with
            class 3 will be kept in the sc_build table. If None then the
            aggregation will be run for each class found in the pras output
            file
        timezone : str
            Timezone for output generation profiles. This is a string that will
            be passed to pytz.timezone() e.g. US/Pacific, US/Mountain,
            US/Central, US/Eastern, or UTC. For a list of all available
            timezones, see pytz.all_timezones
        dset_tag : str
            Dataset tag to append to dataset names in cf profile file. e.g. If
            the cf profile file is a multi year file using dset_tag="-2008"
            will enable us to select the corresponding datasets
            (cf_mean-2008, cf_profile-2008, etc)
        max_workers : int | None
            Max workers for parallel profile aggregation. None uses all
            available workers. 1 will run in serial.
        """

        if res_class is None:
            res_classes = cls.get_pras_classes(pras_file, tech)
        else:
            res_classes = [res_class]
        for res_class in res_classes:
            pa = cls(plexos_nodes, rev_sc_fp, reeds_build_fp, cf_fp,
                     pras_file, forecast_fpath=forecast_fpath,
                     build_year=build_year,
                     plexos_columns=plexos_columns,
                     force_full_build=force_full_build,
                     force_shape_map=force_shape_map,
                     plant_name_col=plant_name_col,
                     tech_tag=tech_tag,
                     tech_type=tech,
                     res_class=res_class,
                     timezone=timezone,
                     dset_tag=dset_tag,
                     max_workers=max_workers)
            missing_nodes += pa.missing_nodes
            profiles = pa.make_profiles()

            pa.export(pa.plexos_meta, pa.time_index, profiles, pras_file)
