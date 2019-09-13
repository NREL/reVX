# -*- coding: utf-8 -*-
"""
Distribute PV to PLEXOS methods.
"""
import shutil
import os
import logging
from warnings import warn
import numpy as np

from reV.handlers.outputs import Outputs

logger = logging.getLogger(__name__)


class DpvResource:
    """Framework to handle and merge multiple reV distributed PV runs."""

    def __init__(self, root_dir, sub_dirs, year):
        """
        Parameters
        ----------
        root_dir : str
            Root directory containing sub directories containing reV dpv job
            files.
        sub_dirs : list
            List of strings of directory names in root_dir. Each sub_dir is a
            job directory.
        year : int | str
            Year of interest.
        """

        self._root_dir = root_dir
        self._sub_dirs = sub_dirs
        self._year = year
        self._fpaths = self._get_all_fpaths()

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

        if not sub_dir.endswith('/'):
            sub_dir += '/'
        d = os.path.join(self._root_dir, sub_dir)
        for fn in os.listdir(d):
            if (fn.endswith('.h5') and str(self._year) in fn
                    and 'node' not in fn):
                fpath = os.path.join(d, fn)
                break
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

    @classmethod
    def merge(cls, root_dir, sub_dirs, year, fn_out, job_frac_map):
        """Merge multiple DPV reV runs into a single result based on fractional
        contributions.

        Parameters
        ----------
        root_dir : str
            Root directory containing sub directories containing reV dpv job
            files.
        sub_dirs : list
            List of strings of directory names in root_dir. Each sub_dir is a
            job directory.
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
        dpv = cls(root_dir, sub_dirs, year)
        fpath_out = os.path.join(dpv._root_dir, fn_out)
        shutil.copy(dpv._fpaths[0], fpath_out)
        with Outputs(fpath_out) as out:
            dsets = [d for d in out.dsets if d not in ['time_index', 'meta']]

        for dset in dsets:
            arr = dpv._merge_data(dset, job_frac_map)
            logger.debug('Writing "{}" to merged DPV output file.'
                         .format(dset))
            with Outputs(fpath_out) as out:
                out[dset] = arr


if __name__ == '__main__':
    from reV.utilities.loggers import init_logger
    init_logger(__name__, log_level='DEBUG')
    root_dir = '/projects/naris/generation_data/60min/naris_rev_pv_dgen/'
    sub_dirs = ['dgen_a90_t28', 'dgen_a135_t28', 'dgen_a180_t28',
                'dgen_a225_t28', 'dgen_a270_t28', 'dgen_t0']
    fn_out = 'naris_rev_dpv_{}.h5'

    job_frac_map = {'dgen_a90_t28': 0.13,
                    'dgen_a135_t28': 0.09,
                    'dgen_a180_t28': 0.25,
                    'dgen_a225_t28': 0.09,
                    'dgen_a270_t28': 0.13,
                    'dgen_t0': 0.31}
    year = 2007
    DpvResource.merge(root_dir, sub_dirs, year, fn_out.format(year),
                      job_frac_map)
