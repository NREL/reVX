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

from reVX.plexos.rev_reeds_plexos import PlexosAggregation
from rex import Outputs


logger = logging.getLogger(__name__)


class PrasAggregation(PlexosAggregation):
    """
    Framework to aggregate reV gen profiles to PRAS node power profiles.
    This class takes as input the plexos nodes meta data (lat/lon or shape
    files), rev supply curve table, and reeds buildout table (specifying
    which rev sc points were built and at what capacity). The class
    will build power profiles for each supply curve point and then aggregate
    the sc point profiles to the nearest neighbor pras node (if pras nodes
    are defined by lat/lon) or the shape intersect pras node (if pras nodes
    are defined by shape file).
    """

    def __init__(self, plexos_nodes, rev_sc, reeds_build, cf_fpath, pras_file,
                 forecast_fpath=None, build_year=2050, plexos_columns=None,
                 force_full_build=False, force_shape_map=False,
                 plant_name_col=None, tech_tag=None, res_class=None,
                 tech_type=None, timezone='UTC', max_workers=None):
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
        max_workers : int | None
            Max workers for parallel profile aggregation. None uses all
            available workers. 1 will run in serial.
        """
        super().__init__(plexos_nodes, rev_sc, reeds_build, cf_fpath,
                         forecast_fpath=forecast_fpath,
                         build_year=build_year,
                         plexos_columns=plexos_columns,
                         force_full_build=force_full_build,
                         force_shape_map=force_shape_map,
                         plant_name_col=plant_name_col,
                         tech_tag=tech_tag, res_class=res_class,
                         timezone=timezone, max_workers=max_workers)
        self._tech_type = tech_type
        self._pras_file = pras_file
        self._pras_meta = None
        with Outputs(self._pras_file, mode='r') as out:
            self._pras_meta = pd.DataFrame(out['generators/_core'])
        self._pras_meta = self.enforce_tech_constraint()
        self.check_missing_nodes()

    @property
    def pras_meta(self):
        """
        Get pras 'meta' which describes the tech type and class at each node

        Returns
        -------
        pd.DataFrame
        """
        return self._pras_meta

    @property
    def pras_nodes(self):
        """
        Get list of available nodes in pras output file

        Returns
        -------
        List
        """
        pras_nodes = [int(x.decode('utf-8').replace('p', '')) - 1
                      for x in self.pras_meta['region']]
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
        node_mask = np.isin(self.pras_nodes, self.sc_nodes)
        pras_indices = self.pras_meta.index[node_mask]
        return pras_indices

    @property
    def profile_indices(self):
        """
        Get the list of generation profile indices used to write the new
        profiles to the pras output.

        Returns
        -------
        List
        """
        profile_indices = [i for i, x in enumerate(self.sc_nodes)
                           if x in self.pras_nodes]
        return profile_indices

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
        category_mask = [x.decode('utf-8').startswith(tech_key)
                         for x in self._pras_meta['category']]
        self._pras_meta = self._pras_meta[category_mask]
        return self._pras_meta

    def check_missing_nodes(self):
        """
        Check for nodes present in the pras output file which cannot be found
        in the supply curve node list
        """
        node_mask = np.isin(self.pras_nodes, self.sc_nodes)
        missing_nodes = sorted([x for x in np.unique(self.pras_nodes)
                                if x not in self.sc_nodes])
        if not all(node_mask):
            msg = ('Unable to find requested tech type {}_{} for {} out of {} '
                   'pras nodes: {}'
                   .format(self._tech_type, self._res_class,
                           len(missing_nodes), len(np.unique(self.pras_nodes)),
                           missing_nodes))
            logger.warning(msg)
            warn(msg)

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
        out_fpath : str, optional
            Path to .h5 file into which plant buildout should be saved. A
            pras-formatted csv will also be written in the same directory.
            By default None.
        """
        if not out_fpath.endswith('.h5'):
            out_fpath = out_fpath + '.h5'

        msg = ("out_fpath={} does not exist. This file must come from a "
               "previous ReEDS2PRAS run").format(out_fpath)
        assert os.path.exists(out_fpath), msg

        shutil.copy(out_fpath,
                    out_fpath.replace('.h5', f'_{self.tz_alias}.h5'))
        out_fpath = out_fpath.replace('.h5', f'_{self.tz_alias}.h5')

        logger.info('Saving result to file: {}'.format(out_fpath))

        profiles = self.tz_convert_profiles(profiles, self._timezone)

        with Outputs(out_fpath, mode='a') as out:
            out['generators/capacity'][:, self.pras_indices] = \
                profiles[:, self.profile_indices]

        names = np.arange(profiles.shape[1])
        if self._plant_name_col is not None:
            names = self.get_unique_plant_names(meta, self._plant_name_col,
                                                self._tech_tag)

        df_plx = pd.DataFrame(profiles, columns=names,
                              index=time_index.tz_convert(None))
        df_plx.index.name = 'DATETIME'
        csv_fp = out_fpath.replace('.h5', '.csv')
        df_plx.to_csv(csv_fp)

        logger.info('Wrote pras formatted profiles to: {}'.format(csv_fp))
