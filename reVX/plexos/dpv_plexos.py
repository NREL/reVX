# -*- coding: utf-8 -*-
"""
Distribute PV to PLEXOS methods.
"""
import shutil
import os
import logging
from warnings import warn
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from reVX.handlers.outputs import Outputs

logger = logging.getLogger(__name__)


class DpvResource:
    """Framework to handle and merge multiple reV distributed PV runs."""

    def __init__(self, root_dir, year, sub_dirs=None):
        """
        Parameters
        ----------
        root_dir : str
            Root directory containing sub directories containing reV dpv job
            files.
        year : int | str
            Year of interest.
        sub_dirs : list
            List of strings of directory names in root_dir. Each sub_dir is a
            job directory.
        """

        self._root_dir = root_dir
        self._year = year
        self._sub_dirs = self._parse_sub_dirs(self._root_dir,
                                              sub_dirs=sub_dirs)
        self._fpaths = self._get_all_fpaths()

    @staticmethod
    def _parse_sub_dirs(root_dir, sub_dirs=None):
        """Get a list of strings of directory names in root_dir.

        Parameters
        ----------
        root_dir : str
            Root directory containing sub directories containing reV dpv job
            files.
        sub_dirs : list | None
            List of strings of directory names in root_dir. Each sub_dir is a
            job directory. None to parse dirs.

        Returns
        -------
        sub_dirs : list
            List of strings of directory names in root_dir. Each sub_dir is a
            job directory.
        """

        if sub_dirs is None:
            sub_dirs = os.listdir(root_dir)
            sub_dirs = [s for s in sub_dirs if
                        os.path.isdir(os.path.join(root_dir, s))]
        return sub_dirs

    def _get_sub_dir_fpath(self, sub_dir):
        """Get the full h5 filepath of the job in the target sub dir.

        Parameters
        ----------
        sub_dir : str
            Directory name in root_dir.

        Returns
        -------
        fpath : str
            Full filepath to the h5 job file in root_dir/sub_dir/
        """

        fpaths = []

        d = os.path.join(self._root_dir, sub_dir)

        for fn in os.listdir(d):
            if (fn.endswith('.h5') and str(self._year) in fn
                    and 'node' not in fn):
                fpaths.append(os.path.join(d, fn))

        fpath = fpaths[0]

        if len(fpaths) > 1:
            wmsg = 'More than one h5 filepath found. Using: {}'.format(fpath)
            logger.warning(wmsg)
            warn(wmsg)

        return fpath

    def _get_all_fpaths(self):
        """Get a list of all job filepaths in all sub directories.

        Returns
        -------
        fpaths : list
            List of full filepaths to h5 job files in root_dir/sub_dir/
        """

        fpaths = []
        for sub_dir in self._sub_dirs:
            fpaths.append(self._get_sub_dir_fpath(sub_dir))
        return fpaths

    def _merge_data(self, dset, job_frac_map):
        """Merge reV data from multiple dpv jobs based on relative fractions.

        Example: batch job to run multiple DPV jobs at different tilt/azimuths,
        this method will combine all of the different jobs based on respective
        fractions in the job_frac_map into a single composite DPV result.

        Parameters
        ----------
        dset : str
            Dataset name to merge (i.e. cf_mean, cf_profile, etc...)
        job_frac_map : dict
            Lookup of the fractional contribution for each job.
            The keys are job names (string tags found in the sub_dirs and
            job h5 files), and the values are fractions for the
            respective jobs.

        Returns
        -------
        arr : np.ndarray
            Unscaled numpy array of dset data merged from all fpaths based
            on the job fractions.
        """
        arr = None
        total_frac = 0
        for fpath in self._fpaths:
            apply_frac = None
            for job_tag, frac in job_frac_map.items():
                if job_tag in os.path.basename(fpath):
                    apply_frac = frac
                    total_frac += frac
                    break

            if apply_frac is not None:
                logger.debug('Merging "{}" with frac {} from: {}'
                             .format(dset, apply_frac,
                                     os.path.basename(fpath)))
                if arr is None:
                    with Outputs(fpath) as out:
                        arr = (apply_frac * out[dset])
                else:
                    with Outputs(fpath) as out:
                        arr += (apply_frac * out[dset])

        if np.abs(total_frac - 1) > 0.01:
            wmsg = ('Fractions sum to {} (should be 1.00) when merging '
                    'DPV jobs: {}'.format(total_frac, self._fpaths))
            warn(wmsg)
            logger.warning(wmsg)

        return arr

    def _merge_dpv_files(self, fn_out, job_frac_map):
        """Merge multiple DPV reV runs into a single result based on fractional
        contributions.

        Parameters
        ----------
        fn_out : str
            Filename for merged results, to be saved in root_dir.
        job_frac_map : dict
            Lookup of the fractional contribution for each job.
            The keys are job names (string tags found in the sub_dirs and
            job h5 files), and the values are fractions for the
            respective jobs.
        """

        fpath_out = os.path.join(self._root_dir, fn_out)
        shutil.copy(self._fpaths[0], fpath_out)
        with Outputs(fpath_out) as out:
            dsets = [d for d in out.datasets
                     if d not in ['time_index', 'meta']]

        for dset in dsets:
            arr = self._merge_data(dset, job_frac_map)
            logger.debug('Writing "{}" to merged DPV output file.'
                         .format(dset))
            with Outputs(fpath_out, mode='a') as out:
                out[dset] = arr

    @classmethod
    def merge(cls, root_dir, year, fn_out, job_frac_map):
        """Merge multiple DPV reV runs into a single result based on fractional
        contributions.

        Parameters
        ----------
        root_dir : str
            Root directory containing sub directories containing reV dpv job
            files.
        year : int | str
            Year of interest.
        fn_out : str
            Filename for merged results, to be saved in root_dir.
        job_frac_map : dict
            Lookup of the fractional contribution for each job.
            The keys are job names (string tags found in the sub_dirs and
            job h5 files), and the values are fractions for the
            respective jobs.
        """

        dpv = cls(root_dir, year, sub_dirs=list(job_frac_map.keys()))
        dpv._merge_dpv_files(fn_out, job_frac_map)


class DpvPlexosAggregation:
    """Methods to aggregate DPV profiles to plexos nodes."""

    def __init__(self, node_buildout, cf_fpath):
        """
        Parameters
        ----------
        node_buildout : str | pd.DataFrame
            Plexos node buildout meta data. Must have the following columns:
            [plexos_id, latitude, longitude, built_capacity]
        cf_fpath : str
            Filepath to a reV capacity factor file.
        """
        logger.info('Starting DPV to PLEXOS aggregation.')

        self._parse_node_buildout(node_buildout)
        self._cf_fpath = cf_fpath
        self._cf_meta = None
        self._time_index = None
        self._kdtree = None

        logger.debug('Using CF file: {}'.format(self._cf_fpath))

    def _parse_node_buildout(self, node_buildout,
                             req=('plexos_id', 'latitude', 'longitude',
                                  'built_capacity')):
        """Parse the node buildout table.

        Parameters
        ----------
        node_buildout : str | pd.DataFrame
            Plexos node buildout meta data.
        req : list
            List of required column names in node_buildout
        """

        if isinstance(node_buildout, str):
            logger.debug('Using PLEXOS node file: {}'.format(node_buildout))
            self._node_buildout = pd.read_csv(node_buildout)
        elif isinstance(node_buildout, pd.DataFrame):
            self._node_buildout = node_buildout
        else:
            raise TypeError('Did not recognize the type of the node buildout '
                            'table input: {}'.format(type(node_buildout)))
        missing = []
        for r in req:
            if r not in self._node_buildout:
                missing.append(r)
        if any(missing):
            raise KeyError('Node buildout table is missing the following '
                           'columns: {}'.format(missing))

        # drop node buildouts without capacity
        bad = (pd.isnull(self._node_buildout['built_capacity'])
               | (self._node_buildout['built_capacity'] == 0))
        bad = self._node_buildout.index.values[bad]
        self._node_buildout = self._node_buildout.drop(bad, axis=0)
        self._node_buildout = self._node_buildout.reset_index(drop=True)

    @property
    def cf_meta(self):
        """Get the cf meta data.

        Returns
        -------
        _cf_meta : pd.DataFrame
            reV CF Meta data.
        """
        if self._cf_meta is None:
            with Outputs(self._cf_fpath) as out:
                self._cf_meta = out.meta
        return self._cf_meta

    @property
    def time_index(self):
        """Get the cf time index.

        Returns
        -------
        _time_index : pd.DateTimeindex
            reV timeindex data.
        """
        if self._time_index is None:
            with Outputs(self._cf_fpath) as out:
                self._time_index = out.time_index
        return self._time_index

    @property
    def kdtree(self):
        """Get the KDtree of the reV CF meta data coordinates.

        Returns
        -------
        _kdtree : scipy.spatial.ckdtree
            KDtree build from the reV CF meta data coordinates.
        """
        if self._kdtree is None:
            self._kdtree = cKDTree(self.cf_meta[['latitude', 'longitude']])
        return self._kdtree

    def _run_nn(self, query):
        """Run KDTree query, getting CF index values matching the node buildout

        Parameters
        ----------
        query : np.ndarray
            Query the kdtree attribute with this argument.

        Returns
        -------
        i : np.ndarray
            1D array of CF index values.
        """

        d, i = self.kdtree.query(query, k=1)
        logger.info('KDTree distance min / mean / max: {} / {} / {}'
                    .format(np.round(d.min(), decimals=3),
                            np.round(d.mean(), decimals=3),
                            np.round(d.max(), decimals=3)))
        return i

    def get_mapped_node_data(self, dset='cf_profile'):
        """Get the CF dataset mapped to plexos nodes via NN results.

        Returns
        -------
        node_arr : np.ndarray
            Get a 2D dataset from the reV cf file mapped to the node_buildout
            table via NN.
        """
        with Outputs(self._cf_fpath) as out:
            arr = out[dset]
        i = self._run_nn(self._node_buildout[['latitude', 'longitude']])
        node_arr = arr[:, i]
        return node_arr

    def get_node_gen_profiles(self):
        """Get the generation profiles (cf * capacity) mapped to the node meta.

        Returns
        -------
        gen_profiles : np.ndarray
            Timeseries profiles of (t, n) where t is the time index and n is
            the node buildout. Units are the built_capacity units from the
            node_buildout table.
        """
        node_arr = self.get_mapped_node_data(dset='cf_profile')
        built_capacity = self._node_buildout['built_capacity'].values
        gen_profiles = node_arr * built_capacity
        return gen_profiles

    @classmethod
    def run(cls, node_buildout, cf_fpath, out_fpath):
        """Aggregate dpv reV gen data to a buildout nodes and save to h5.

        Parameters
        ----------
        node_buildout : str | pd.DataFrame
            Plexos node buildout meta data. Must have the following columns:
            [plexos_id, latitude, longitude, built_capacity]
        cf_fpath : str
            Filepath to a reV capacity factor file.
        out_fpath : str
            Output filepath.
        """

        pa = cls(node_buildout, cf_fpath)
        profiles = pa.get_node_gen_profiles()

        logger.info('Writing results to file: {}'.format(out_fpath))
        with Outputs(out_fpath, mode='w') as out:
            meta = out.to_records_array(pa._node_buildout)
            time_index = np.array(pa.time_index.astype(str), dtype='S20')
            out._create_dset('dpv/meta', meta.shape, meta.dtype, data=meta)
            out._create_dset('dpv/time_index', time_index.shape,
                             time_index.dtype, data=time_index)
            out._create_dset('dpv/gen_profiles', profiles.shape,
                             profiles.dtype, chunks=(None, 100),
                             data=profiles)

        logger.info('Finished DPV to PLEXOS aggregation.')
