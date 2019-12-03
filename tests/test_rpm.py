# -*- coding: utf-8 -*-
"""reVX PLEXOS unit test module
"""
import os
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from reV import TESTDATADIR as reV_TESTDATADIR
from reVX import TESTDATADIR as reVX_TESTDATADIR
from reVX.rpm.rpm_manager import RPMClusterManager


JOB_TAG = 'pytest'
CF_FPATH = os.path.join(reV_TESTDATADIR,
                        'gen_out/gen_ri_pv_2012_x000.h5')
EXCL_FPATH = os.path.join(reV_TESTDATADIR,
                          'ri_exclusions/ri_exclusions.h5')
OUT_DIR = os.path.join(reVX_TESTDATADIR, 'rpm/test_outputs')
TECHMAP_DSET = 'techmap_nsrdb_ri_truth'
EXCL_DICT = {'ri_srtm_slope': {'inclusion_range': (None, 5)},
             'ri_padus': {'exclude_values': [1]}}
RPM_META = os.path.join(reVX_TESTDATADIR, 'rpm/rpm_meta.csv')
BASELINE = os.path.join(reVX_TESTDATADIR,
                        'rpm/rpm_cluster_outputs_baseline.csv')
TEST = os.path.join(OUT_DIR, 'rpm_cluster_outputs_{}.csv'.format(JOB_TAG))

PURGE_OUT = True


def test_rpm():
    """Test the rpm clustering pipeline and run a baseline validation."""

    RPMClusterManager.run_clusters_and_profiles(CF_FPATH, RPM_META, EXCL_FPATH,
                                                EXCL_DICT, TECHMAP_DSET,
                                                OUT_DIR, job_tag=JOB_TAG,
                                                rpm_region_col=None,
                                                parallel=True,
                                                output_kwargs=None,
                                                dist_rank_filter=True,
                                                contiguous_filter=False)

    df_baseline = pd.read_csv(BASELINE)
    df_test = pd.read_csv(TEST)

    assert_frame_equal(df_baseline, df_test)

    if PURGE_OUT:
        for fn in os.listdir(OUT_DIR):
            os.remove(os.path.join(OUT_DIR, fn))


def test_rpm_serial():
    """Test the rpm clustering pipeline in SERIAL and run a baseline
    validation."""

    RPMClusterManager.run_clusters_and_profiles(CF_FPATH, RPM_META, EXCL_FPATH,
                                                EXCL_DICT, TECHMAP_DSET,
                                                OUT_DIR, job_tag=JOB_TAG,
                                                rpm_region_col=None,
                                                parallel=False,
                                                output_kwargs=None,
                                                dist_rank_filter=True,
                                                contiguous_filter=False)

    df_baseline = pd.read_csv(BASELINE)
    df_test = pd.read_csv(TEST)

    assert_frame_equal(df_baseline, df_test)

    if PURGE_OUT:
        for fn in os.listdir(OUT_DIR):
            os.remove(os.path.join(OUT_DIR, fn))


def execute_pytest(capture='all', flags='-rapP'):
    """Execute module as pytest with detailed summary report.

    Parameters
    ----------
    capture : str
        Log or stdout/stderr capture option. ex: log (only logger),
        all (includes stdout/stderr)
    flags : str
        Which tests to show logs and results for.
    """

    fname = os.path.basename(__file__)
    pytest.main(['-q', '--show-capture={}'.format(capture), fname, flags])


if __name__ == '__main__':
    execute_pytest()
