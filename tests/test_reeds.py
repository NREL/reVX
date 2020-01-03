# -*- coding: utf-8 -*-
"""
reVX PLEXOS unit test module
"""
import numpy as np
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from reV.handlers.resource import Resource
from reV.handlers.outputs import Outputs

from reVX.reeds.reeds_classification import ReedsClassifier
from reVX.reeds.reeds_profiles import ReedsProfiles
from reVX.reeds.reeds_timeslices import ReedsTimeslices
from reVX import TESTDATADIR

ROOT_DIR = os.path.join(TESTDATADIR, 'reeds')


def extract_profiles(profiles_h5):
    """
    Extract 'truth' profiles

    Parameters
    ----------
    profiles_h5 : str
        .h5 file containing 'truth' profiles

    Returns
    -------
    profiles : dict
        Dictionary of representative profiles
    """
    profiles = {}
    meta = None
    time_index = None
    with Resource(profiles_h5) as f:
        for ds in f.dsets:
            if 'profile' in ds:
                n = int(ds.split('_')[-1])
                profiles[n] = f[ds]
            elif 'meta' in ds:
                meta = f[ds]
            elif 'time_index' in ds:
                time_index = f[ds]

    return profiles, meta, time_index


def test_classifier():
    """
    Test ReedsClassifier
    """
    path = os.path.join(ROOT_DIR, 'ReEDS_Classifications.csv')
    truth_table_full = pd.read_csv(path)
    path = os.path.join(ROOT_DIR, 'ReEDS_Classifications_Slim.csv')
    truth_table = pd.read_csv(path)
    path = os.path.join(ROOT_DIR, 'ReEDS_Aggregation.csv')
    truth_agg = pd.read_csv(path)

    rev_table = os.path.join(TESTDATADIR, 'reV_sc', 'sc_table.csv')
    resource_classes = os.path.join(ROOT_DIR, 'inputs',
                                    'trg_breakpoints_naris.csv')
    kwargs = {'cluster_on': 'trans_cap_cost', 'method': 'kmeans'}
    out = ReedsClassifier.create(rev_table, resource_classes,
                                 region_map='reeds_region',
                                 sc_bins=5, cluster_kwargs=kwargs)
    test_table_full, test_table, test_agg = out
    assert_frame_equal(truth_table_full, test_table_full, check_dtype=False,
                       check_exact=False)
    assert_frame_equal(truth_table, test_table, check_dtype=False,
                       check_exact=False)
    assert_frame_equal(truth_agg, test_agg, check_dtype=False,
                       check_exact=False)


def test_profiles():
    """
    Test ReedsProfiles
    """
    truth = extract_profiles(os.path.join(ROOT_DIR, 'ReEDS_Profiles.h5'))

    cf_profiles = os.path.join(TESTDATADIR, 'reV_gen', 'gen_pv_2012.h5')
    rev_table = os.path.join(ROOT_DIR, 'ReEDS_Classifications.csv')
    test = ReedsProfiles.run(cf_profiles, rev_table,
                             profiles_dset='cf_profile', rep_method='meanoid',
                             err_method='rmse', n_profiles=3,
                             reg_cols=('region', 'class'),
                             parallel=False)
    for k, v in truth[0].items():
        msg = 'Representative profiles {} do not match!'.format(k)
        assert np.allclose(v, test[0][k]), msg

    assert_frame_equal(truth[1], test[1], check_dtype=False, check_exact=False)
    assert truth[2].equals(test[2]), 'time_index does not match!'


def test_rep_timeslices():
    """
    Test ReedsTimeslices from representative profiles
    """
    rep_profiles = os.path.join(ROOT_DIR, 'ReEDS_Profiles.h5')
    timeslice_map = os.path.join(ROOT_DIR, 'inputs',
                                 'timeslices.csv')
    test_stats, test_coeffs = ReedsTimeslices.run(rep_profiles, timeslice_map,
                                                  max_workers=1,
                                                  legacy_format=True)

    path = os.path.join(ROOT_DIR, 'ReEDS_Timeslice_rep_stats.csv')
    truth = pd.read_csv(path)
    assert_frame_equal(truth, test_stats, check_dtype=False,
                       check_exact=False, check_less_precise=True)

    path = os.path.join(ROOT_DIR, 'ReEDS_Timeslice_rep_coeffs.csv')
    truth = pd.read_csv(path)
    assert_frame_equal(truth, test_coeffs, check_dtype=False,
                       check_exact=False, check_less_precise=True)


def test_cf_timeslices():
    """
    Test ReedsTimeslices from CF profiles
    """
    cf_profiles = os.path.join(TESTDATADIR, 'reV_gen', 'gen_pv_2012.h5')
    rev_table = os.path.join(ROOT_DIR, 'ReEDS_Classifications.csv')
    timeslice_map = os.path.join(ROOT_DIR, 'inputs',
                                 'timeslices.csv')
    test, _ = ReedsTimeslices.run(cf_profiles, timeslice_map,
                                  rev_table=rev_table, max_workers=1,
                                  legacy_format=True)

    path = os.path.join(ROOT_DIR, 'ReEDS_Timeslice_cf_stats.csv')
    truth = pd.read_csv(path)
    assert_frame_equal(truth, test, check_dtype=False,
                       check_exact=False, check_less_precise=True)


def test_timeslice_h5_output():
    """
    Test h5 output for timeslice correlation table.
    """

    rep_profiles = os.path.join(ROOT_DIR, 'ReEDS_Profiles.h5')
    timeslice_map = os.path.join(ROOT_DIR, 'inputs',
                                 'timeslices.csv')
    reg_cols = ('region', 'class')
    fpath = os.path.join(ROOT_DIR, 'ReEDS_Corr_Coeffs.h5')
    _, test_coeffs = ReedsTimeslices.run(rep_profiles, timeslice_map,
                                         max_workers=1,
                                         legacy_format=False,
                                         reg_cols=reg_cols)

    ReedsTimeslices.save_correlation_dict(test_coeffs, reg_cols, fpath)

    with Outputs(fpath) as out:
        meta = out.meta
        assert len(out.dsets) == len(test_coeffs) + 1
        for k, v in test_coeffs.items():
            dset = 'timeslice_{}'.format(k)
            data = out[dset]
            assert np.allclose(data, np.round(v, decimals=3), atol=0.001)
            assert len(meta) == len(data)
    os.remove(fpath)


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
