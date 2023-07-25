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
from rex import Outputs


logger = logging.getLogger(__name__)


class PrasAggregation(PlexosAggregation):
    """
    Framework to aggregate reV gen profiles to PRAS zone power profiles and
    overwrite exisiting pras file with new generation profiles. This class
    takes as input rev supply curve table, and reeds buildout table (specifying
    which rev sc points were built and at what capacity). The class will build
    power profiles for each supply curve point and then aggregate the sc point
    profiles to the nearest neighbor pras zone. The corresponding zone power
    profiles in the pras output file will be overwritten with these new
    aggregated generation profiles.

    Examples
    --------

    >>> from reVX.pras.rev_reeds_pras import PrasAggregation
    >>>
    >>> rev_supply_curve_file = ...
    >>> reeds_build_file = ...
    >>> capacity_factor_file = ...
    >>> pras_file = ...
    >>>
    >>> PrasAggregation.run(rev_supply_curve_file, reeds_build_file,
                            capacity_factor_file, pras_file, build_year=2047,
                            tech_type='upv')

    This will replace all the pras generation profiles with upv tech in the
    file pras_file.pras.h5 with aggregated generation profiles from cf_path.h5

    >>> reeds_build_files = [...]
    >>> capacity_factor_files = [...]
    >>> tech_types = ['upv', 'dupv', 'wind-ons', 'wind-ofs']
    >>>
    >>> PrasAggregation.run(rev_supply_curve_file, reeds_build_files,
                            capacity_factor_files, pras_file, build_year=2047,
                            tech_type=tech_types)

    This will run the same routine but for each tech type in tech_types. The
    files in reeds_build_files and capacity_factor_files have to correspond to
    the tech type in the tech_types list.
    """
    def __init__(self, rev_sc, reeds_build, cf_fpath, pras_file,
                 forecast_fpath=None, build_year=2050, res_class=None,
                 tech_type=None, timezone='US/Central', dset_tag=None,
                 max_workers=None):
        """
        Parameters
        ----------
        rev_sc : str | pd.DataFrame
            reV supply curve results table including SC gid, latitude,
            longitude, res_gids, gid_counts. Or file path to reV supply
            curve table.
        reeds_build : str | pd.DataFrame
            ReEDS buildout with rows for built capacity (MW) at each reV SC
            point. This should have columns: year, region, class,
            built_capacity, and sc_gid (corresponding to the reV supply curve
            point gid). Some cleaning of the column names will be performed for
            legacy tables but these are the column headers that are desired.
        cf_fpath : str
            File path to .h5 capacity factor file (reV gen output) to get
            profiles from.
        pras_file : str
            File path to existing hdf5 PRAS file (.pras or .h5 extension) which
            will be amended to include generation data from given resource
            data. This file must have a 'generators' group with '_core' and
            'capacity' datasets in this group.
        forecast_fpath : str | None
            Forecasted capacity factor .h5 file path (reV results). If not
            None, the supply curve res_gids are mapped to sites in the
            cf_fpath, then the coordinates from cf_fpath are mapped to the
            nearest neighbor sites in the forecast_fpath, where the final
            generation profiles are retrieved from.
        build_year : int, optional
            REEDS year of interest, by default 2050
        res_class : int | None
            Optional resource class to use to filter supply curve points.
            For example, if res_class = 3 then only supply curve points with
            class 3 will be kept in the sc_build table. The corresponds to the
            'class' column in the reeds_build file.
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
        self._pras_file = pras_file
        self._output_file = pras_file
        self._pras_meta = None
        self._res_class = res_class
        self._complete_pras_meta = None
        self._build_year = build_year
        self._sc_build_indices = None
        self._pras_indices = None
        self._qa_file = None
        self._prof_file = None
        self._pras_build_zones = self._init_pras_build_zones(reeds_build)
        super().__init__(self._pras_build_zones, rev_sc,
                         reeds_build, cf_fpath,
                         forecast_fpath=forecast_fpath,
                         build_year=build_year,
                         res_class=res_class,
                         timezone=timezone,
                         dset_tag=self._dset_tag,
                         max_workers=max_workers)
        self._init_output_files(pras_file)
        logger.info('Running aggregation for tech_type={}, res_class={}, '
                    'rev_sc={}, reeds_build={}, cf_fpath={}, dset_tag={}, '
                    'pras_file={}.'
                    .format(self._tech_type, self._res_class, rev_sc,
                            reeds_build, cf_fpath, dset_tag, pras_file))
        logger.info('Found {} pras_zones and {} sc_build_zones'
                    .format(len(self.pras_zones), len(self.sc_build_zones)))
        self.missing_zones = self.get_missing_zones()
        self.found_zones = list(set(self.pras_zones) - set(self.missing_zones))

    def _init_pras_build_zones(self, reeds_build):
        """
        Build pras_build_zones table from reeds_build. This returns a table
        with all available zones in the reeds build table and assigns a
        plexos_id to each zone so that this can be used as a plexos node table
        for calling super().__init__()

        Parameters
        ----------
        reeds_build : str | pd.DataFrame
            ReEDS buildout with rows for built capacity (MW) at each reV SC
            point. This should have columns: reeds_year, built_capacity, and
            sc_gid (corresponding to the reV supply curve point gid). Some
            cleaning of the column names will be performed for legacy tables
            but these are the column headers that are desired.

        Returns
        -------
        pras_build_zones : pd.DataFrame
            pras zones DataFrame with all available zones in reeds build table
            and corresponding plexos_id
        """
        pras_build_zones = reeds_build
        if isinstance(pras_build_zones, str):
            pras_build_zones = pd.read_csv(pras_build_zones)
        class_mask = pras_build_zones['class'] == self._res_class
        pras_build_zones = pras_build_zones[class_mask]
        year_mask = pras_build_zones['year'] == self._build_year
        pras_build_zones = pras_build_zones[year_mask]
        pras_build_zones = pras_build_zones[['latitude', 'longitude',
                                             'region', 'built_capacity']]
        pras_build_zones_grp = pras_build_zones.groupby(by='region')
        pras_build_zones = pras_build_zones_grp.mean()
        cap = pras_build_zones_grp.sum()['built_capacity']
        pras_build_zones['built_capacity'] = cap
        pras_build_zones = pras_build_zones.reset_index()
        pras_build_zones['plexos_id'] = np.arange(len(pras_build_zones))
        return pras_build_zones

    def _init_output_files(self, pras_file):
        """
        Initialize the output file to be written with new generation profiles
        and define the qa file name.

        Parameters
        ----------
        pras_file : str
            Path to original pras file containing generation profiles to be
            updated. The file will be copied to a new path (given by
            _output_file) and then updated with the new profiles
        """
        if not self._pras_file.endswith('.h5'):
            self._output_file += '.h5'
        self._output_file = pras_file.replace('.h5', '_updated.h5')
        msg = ("pras_file={} does not exist. This file must come from a "
               "previous ReEDS2PRAS run").format(self._pras_file)
        assert os.path.exists(self._pras_file), msg

        if not os.path.exists(self._output_file):
            shutil.copy(self._pras_file, self._output_file)

        self._qa_file = self._output_file.replace(
            '.h5', '_qa_{}_{}.csv'.format(self._tech_type, self._res_class))

        self._prof_file = self._output_file.replace(
            '.h5', '_{}_{}.csv'.format(self._tech_type, self._res_class))

    @classmethod
    def get_pras_meta(cls, pras_file):
        """
        Get pras 'meta' which describes the tech type and class at each zone
        For upv and dupv regions are labeled p1, p2, etc. For wind and csp
        regions are labeled s1, s2, etc.

        Parameters
        ----------
        pras_file : str
            Path to the pras file being overwritten with new generation
            profiles

        Returns
        -------
        pd.DataFrame
        """
        def zone_filter(x):
            out = re.search(r'p\d+', x)
            if out is None:
                out = re.search(r's\d+', x)
            return out[0]

        with Outputs(pras_file, mode='r') as out:
            pras_meta = pd.DataFrame(out['generators/_core'])
            for col in pras_meta.columns:
                pras_meta[col] = pras_meta[col].apply(
                    lambda x: x.decode('utf-8'))

        pras_meta['region'] = pras_meta['name'].apply(zone_filter).values
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
    def zone_index_map(self):
        """
        Get the map between zone index and zone label. i.e. zone_index=1,
        zone_label=p10
        """
        zone_index_map = dict(
            zip(self._pras_build_zones['region'].values,
                self._pras_build_zones['plexos_id'].values))
        return zone_index_map

    @property
    def sc_build_zones(self):
        """
        Get the labels for the zones in the zone_map. For example, the zone_map
        might have an index of 1 for the region but this corresponds to the
        label of p10.
        """
        build_zones = self._pras_build_zones['region'].unique()
        return build_zones

    @property
    def complete_pras_meta(self):
        """
        Get pras 'meta' without filtering for tech_type and class

        Returns
        -------
        pd.DataFrame
        """
        if self._complete_pras_meta is None:
            self._complete_pras_meta = self.get_pras_meta(self._pras_file)
        return self._complete_pras_meta

    @property
    def pras_meta(self):
        """
        Get pras 'meta' which describes the tech type and class at each zone

        Returns
        -------
        pd.DataFrame
        """
        if self._pras_meta is None:
            self._pras_meta = self.enforce_tech_constraint(
                self.complete_pras_meta)
        return self._pras_meta

    @property
    def pras_zones(self):
        """
        Get list of all available zones in pras output file for the requested
        tech and class. For upv and dupv regions are labeled p1, p2, etc. For
        wind and csp regions are labeled s1, s2, etc.

        Returns
        -------
        List
        """
        return self.pras_meta['region'].values

    @property
    def pras_indices(self):
        """
        Get the list of pras file indices for the pras zones available in the
        supply curve zone list.

        Returns
        -------
        np.ndarray
        """
        if self._pras_indices is None:
            zone_mask = np.isin(self.pras_zones, self.sc_build_zones)
            self._pras_indices = list(self.pras_meta.index[zone_mask])
        return self._pras_indices

    @property
    def pras_names(self):
        """
        Get the list of pras entry names for the pras zones available in the
        supply curve zone list.

        Returns
        -------
        np.ndarray
        """
        return self.pras_meta['name'].loc[self.pras_indices]

    @property
    def found_pras_zones(self):
        """
        List of pras zones found in the reeds build out for the requested tech
        and class

        Returns
        -------
        np.ndarray
        """
        return self.pras_meta['region'].loc[self.pras_indices]

    @property
    def found_build_zones(self):
        """
        List of sc build zones found in the pras output file for the requested
        tech and class

        Returns
        -------
        np.ndarray
        """
        return self.sc_build_zones[self.sc_build_indices]

    @property
    def sc_build_indices(self):
        """
        Get the list of profile indices corresponding to the sc_build_zones -
        used to write the new profiles to the pras output.

        Returns
        -------
        List
        """
        if self._sc_build_indices is None:
            zones = self.sc_build_zones.tolist()
            self._sc_build_indices = [zones.index(x) for x
                                      in self.found_pras_zones]
        return self._sc_build_indices

    @property
    def built_capacity(self):
        """
        Get the built capacity for all zones found in the pras output file and
        the reeds build file.

        Returns
        -------
        np.ndarray
        """
        indices = [np.where(self._pras_build_zones['region'] == x)[0][0]
                   for x in self.found_pras_zones]
        cap = self._pras_build_zones['built_capacity'].iloc[indices].values
        return cap

    def enforce_tech_constraint(self, meta):
        """
        Filter the pras generator meta for the requested technology type and
        class

        Parameters
        ----------
        meta : pd.DataFrame
            Full pras 'meta' without tech_type or class filtering

        Returns
        -------
        pd.DataFrame
            pras meta DataFrame filtered by tech_type and class
        """
        tech_key = '{}_'.format(self._tech_type)
        if self._res_class is not None:
            tech_key += '{}'.format(self._res_class)
        category_mask = [x.startswith(tech_key) for x in meta['category']]
        out_meta = meta[category_mask]
        if out_meta.empty:
            msg = ('Found zero pras zones with requested tech_key {}.'
                   .format(tech_key))
            logger.error(msg)
            raise RuntimeError(msg)
        return out_meta

    def get_missing_zones(self):
        """
        Check for zones present in the pras output file which cannot be found
        in the supply curve zone list
        """
        zone_mask = np.isin(self.pras_zones, self.sc_build_zones)
        missing_zones = sorted([x for x in np.unique(self.pras_zones)
                                if x not in self.sc_build_zones])
        msg = ('{} / {} pras zones not found in sc_build_zones, for '
               'tech_key {}_{}. Missing zones: {}.').format(
                   len(missing_zones), len(self.pras_zones), self._tech_type,
                   self._res_class, missing_zones)
        if not all(zone_mask):
            logger.warning(msg)
            warn(msg)
        else:
            logger.info('All {} pras zones found in sc_build_zones'
                        .format(len(self.pras_zones)))

        if len(missing_zones) > 0:
            missing_indices = self.pras_meta['region'].isin(missing_zones)
            missing_indices = self.pras_meta.index[missing_indices].tolist()
            with Outputs(self._pras_file, mode='r') as out:
                missing_cap = out['generators/capacity', :, missing_indices]
                logger.info('The mean/min/max capacity for the missing zones: '
                            '{} / {} / {} MW.'
                            .format(np.mean(missing_cap), np.min(missing_cap),
                                    np.max(missing_cap)))

        return missing_zones

    def qa(self, profiles):
        """
        Simple QA to check original mean/min/max capacity and the mean/min/max
        capacity for the updated generation profiles

        Parameters
        ----------
        profiles : np.ndarray
            Generation profile timeseries in MW at each plant / pras zone.

        Returns
        -------
        new_cap : np.ndarray
            Array containing the updated generation profiles
        """
        df = pd.DataFrame()
        with Outputs(self._pras_file, mode='r') as out:
            old_cap = out['generators/capacity', :, self.pras_indices]
            new_cap = profiles[:, self.sc_build_indices].astype(old_cap.dtype)
            df['region'] = self.found_pras_zones
            df['name'] = self.pras_meta['name'].loc[self.pras_indices]
            df['old_mean'] = np.mean(old_cap, axis=0)
            df['new_mean'] = np.mean(new_cap, axis=0)
            df['old_max'] = np.max(old_cap, axis=0)
            df['new_max'] = np.max(new_cap, axis=0)
            df['built_capacity'] = self.built_capacity
            logger.info('The old and new mean/max capacity (MW) for the {}_{} '
                        'zones:\n{}.'.format(self._tech_type,
                                             self._res_class, df))
            df.to_csv(self._qa_file)
            logger.info('Wrote pras QA to: {}'.format(self._qa_file))

            return new_cap

    def export(self, time_index, profiles):
        """Export generation profiles to h5 and plexos-formatted csv

        Parameters
        ----------
        time_index : pd.datetimeindex
            Time index for the profiles.
        profiles : np.ndarray
            Generation profile timeseries in MW at each plant / pras zone.
        """
        logger.info('Saving profiles for {}_{} to file: {}'
                    .format(self._tech_type, self._res_class,
                            self._output_file))

        profiles = self.tz_convert_profiles(profiles, self._timezone)

        new_cap = self.qa(profiles)
        with Outputs(self._output_file, mode='a') as out:
            out['generators/capacity', :, self.pras_indices] = new_cap

        df_pras = pd.DataFrame(new_cap, columns=self.pras_names,
                               index=time_index.tz_convert(None))
        df_pras.index.name = 'DATETIME'
        df_pras.to_csv(self._prof_file)
        logger.info('Wrote pras formatted profiles to: {}'
                    .format(self._prof_file))

    def _make_node_map(self):
        """Map built rev SC points to pras zones.

        Returns
        -------
        pras_zone_index : np.ndarray
            (n, 1) array of pras zone indices mapped to the SC builds where n
            is the number of SC points built.  Each value in this array gives
            the pras zone index that the sc point is mapped to. So
            self.zone_map[10] yields the pras zone index for
            self.sc_build[10].
        """
        if 'region' in self.sc_build:
            pras_zone_index = self.sc_build['region'].values
            pras_zone_index = [self.zone_index_map[k] for k
                               in pras_zone_index]
            pras_zone_index = np.array(pras_zone_index)

            if len(pras_zone_index.shape) == 1:
                pras_zone_index = pras_zone_index.reshape(
                    (len(pras_zone_index), 1))
        else:
            msg = '"region" column not found in sc_build.'
            logger.error(msg)
            raise RuntimeError(msg)
        return pras_zone_index

    @classmethod
    def run(cls, rev_sc, reeds_build, cf_fpath, pras_file,
            forecast_fpath=None, build_year=2050, res_class=None,
            tech_type=None, timezone='US/Central', dset_tag=None,
            max_workers=None):
        """Run pras aggregation and output for the requested tech types. This
        will aggregate the generation profiles over region specific supply
        curve points found in rev_sc, after filtering rev_sc for the requested
        tech type, and resource class. These aggregated generation profiles
        will then be written to the pras output file in the location
        corresponding to the region, tech type, and class.

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
            supply curve point buildout to a single pras zone. If included,
            all points must be assigned to pras zones. If providing a list
            of tech types this also needs to be a list of the corresponding
            reeds builout files for each tech type.
        cf_fpath : str | list
            File path to capacity factor file (reV gen output) to
            get profiles from. If providing a list of tech types this also
            needs to be a list of the corresponding capacity factor files.
        pras_file : str
            File path to existing PRAS file which will be amended to include
            generation data from given resource data.
        forecast_fpath : str | None
            Forecasted capacity factor .h5 file path (reV results). If not
            None, the supply curve res_gids are mapped to sites in the
            cf_fpath, then the coordinates from cf_fpath are mapped to the
            nearest neighbor sites in the forecast_fpath, where the final
            generation profiles are retrieved from.
        build_year : int, optional
            REEDS year of interest, by default 2050
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
        missing_zones = []

        for i, tech in enumerate(tech_type):
            reeds_build_fp = reeds_build[i]
            cf_fp = cf_fpath[i]
            rev_sc_fp = rev_sc[i]
            missing_zones += cls._run_single(tech, rev_sc_fp, reeds_build_fp,
                                             cf_fp, pras_file,
                                             forecast_path=forecast_fpath,
                                             build_year=build_year,
                                             res_class=res_class,
                                             timezone=timezone,
                                             dset_tag=dset_tag,
                                             max_workers=max_workers)

        logger.info('Missing {} zones across all classes: {}'
                    .format(len(set(missing_zones)), set(missing_zones)))

    @classmethod
    def _run_single(cls, tech, rev_sc_fp, reeds_build_fp, cf_fp, pras_file,
                    forecast_fpath=None, build_year=2050, res_class=None,
                    timezone='US/Central', dset_tag=None, max_workers=None):
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
            supply curve point buildout to a single pras zone. If included,
            all points must be assigned to pras zones.
        cf_fp : str
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
        build_year : int, optional
            REEDS year of interest, by default 2050
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

        missing_zones = []
        if res_class is None:
            res_classes = cls.get_pras_classes(pras_file, tech)
        else:
            res_classes = [res_class]
        for res_class in res_classes:
            pa = cls(rev_sc_fp, reeds_build_fp, cf_fp, pras_file,
                     forecast_fpath=forecast_fpath,
                     build_year=build_year,
                     tech_type=tech,
                     res_class=res_class,
                     timezone=timezone,
                     dset_tag=dset_tag,
                     max_workers=max_workers)
            profiles = pa.make_profiles()
            pa.export(pa.time_index, profiles)
            missing_zones += pa.missing_zones
        return missing_zones
