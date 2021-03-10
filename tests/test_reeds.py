# -*- coding: utf-8 -*-
"""
reVX ReEDs unit test module
"""
from click.testing import CliRunner
import json
import numpy as np
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import tempfile
import traceback

from rex.resource import Resource
from rex.utilities.loggers import LOGGERS

from reVX.reeds.reeds_cli import main
from reVX.reeds.reeds_classification import ReedsClassifier
from reVX.reeds.reeds_profiles import ReedsProfiles
from reVX.reeds.reeds_timeslices import ReedsTimeslices
from reVX import TESTDATADIR

ROOT_DIR = os.path.join(TESTDATADIR, 'reeds')


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


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
        for ds in f.datasets:
            if 'profile' in ds:
                n = int(ds.split('_')[-1])
                profiles[n] = f[ds]
            elif 'meta' in ds:
                meta = f.meta
            elif 'time_index' in ds:
                time_index = f.time_index

    return profiles, meta, time_index


@pytest.fixture
def bad_trg_classes():
    """
    load TRG classes
    """
    path = os.path.join(TESTDATADIR, 'reeds/inputs/reeds_class_bins.csv')
    trg_classes = pd.read_csv(path)
    trg_classes = trg_classes.loc[trg_classes['sub_type'] == 'fixed']
    return trg_classes[['class', 'TRG_cap', 'mean_res_min']]


@pytest.fixture
def bad_range_classes():
    """
    load TRG classes
    """
    path = os.path.join(TESTDATADIR, 'reeds/inputs/reeds_class_bins.csv')
    trg_classes = pd.read_csv(path)
    trg_classes = trg_classes.loc[trg_classes['sub_type'] == 'fixed']
    return trg_classes[['class', 'mean_res_min', 'sub_type']]


def test_bad_resource_classes(bad_trg_classes, bad_range_classes):
    """
    Ensure ReedsClassifier failes with bad resource_class inputs
    """
    rev_table = os.path.join(TESTDATADIR, 'reV_sc', 'sc_table.csv')

    with pytest.raises(ValueError):
        ReedsClassifier.create(rev_table, bad_trg_classes,
                               region_map='reeds_region',
                               cap_bins=5, sort_bins_by='trans_cap_cost',
                               trg_by_region=True)

    with pytest.raises(ValueError):
        ReedsClassifier.create(rev_table, bad_range_classes,
                               region_map='reeds_region',
                               cap_bins=5, sort_bins_by='trans_cap_cost',
                               trg_by_region=True)


@pytest.fixture
def trg_classes():
    """
    load TRG classes
    """
    path = os.path.join(TESTDATADIR, 'reeds/inputs/reeds_class_bins.csv')
    trg_classes = pd.read_csv(path)
    trg_classes = trg_classes.loc[trg_classes['sub_type'] == 'fixed']
    return trg_classes[['class', 'TRG_cap']]


def test_capacity_bins():
    """
    Test ReedsClassifier._capacity_bins to ensure uniform bin sizes
    """
    rev_table = os.path.join(TESTDATADIR, 'reV_sc', 'sc_table.csv')
    rev_table = pd.read_csv(rev_table)
    rev_table['region'] = 1
    rev_table['class'] = 1

    rev_table = ReedsClassifier._capacity_bins(rev_table, cap_bins=10,
                                               sort_bins_by='trans_cap_cost')

    bin_caps = rev_table.groupby('bin')['capacity'].sum().values
    mean_cap = bin_caps.mean()
    bin_error = np.abs(bin_caps - mean_cap) / mean_cap
    assert np.all(bin_error < 0.2), 'Bin size varies > 20% from the mean'


def test_classifier(trg_classes):
    """
    Test ReedsClassifier
    """
    path = os.path.join(ROOT_DIR, 'ReEDS_Classifications.csv')
    truth_table_full = pd.read_csv(path)
    path = os.path.join(ROOT_DIR, 'ReEDS_Classifications_Slim.csv')
    truth_table = pd.read_csv(path)
    path = os.path.join(ROOT_DIR, 'ReEDS_Aggregation.csv')
    truth_agg = pd.read_csv(path).fillna(0)

    rev_table = os.path.join(TESTDATADIR, 'reV_sc', 'sc_table.csv')
    out = ReedsClassifier.create(rev_table, trg_classes,
                                 region_map='reeds_region',
                                 cap_bins=5, sort_bins_by='trans_cap_cost',
                                 trg_by_region=False)

    test_table_full, test_table, _, test_agg = out

    cols = ['capacity', 'trans_cap_cost', 'dist_mi']
    test_agg[cols] = test_agg[cols].fillna(0)

    assert_frame_equal(truth_table_full, test_table_full, check_dtype=False,
                       check_categorical=False)
    assert_frame_equal(truth_table, test_table, check_dtype=False,
                       check_categorical=False)
    assert_frame_equal(truth_agg, test_agg, check_dtype=False,
                       check_categorical=False)


@pytest.mark.parametrize('max_workers', [None, 1])
def test_profiles(max_workers):
    """
    Test ReedsProfiles
    """
    truth = extract_profiles(os.path.join(ROOT_DIR, 'ReEDS_Profiles.h5'))

    cf_profiles = os.path.join(TESTDATADIR, 'reV_gen', 'gen_pv_2012.h5')
    rev_table = os.path.join(ROOT_DIR, 'ReEDS_Classifications.csv')
    test = ReedsProfiles.run(cf_profiles, rev_table,
                             profiles_dset='cf_profile', rep_method='meanoid',
                             err_method='rmse', n_profiles=3,
                             reg_cols=('region', 'class'), weight='gid_counts',
                             max_workers=max_workers)

    for k, v in truth[0].items():
        msg = 'Representative profiles {} do not match!'.format(k)
        assert np.allclose(v, test[0][k]), msg

    truth[1].index.name = None
    assert_frame_equal(truth[1], test[1], check_dtype=False,
                       check_categorical=False)

    truth_ti = truth[2].tz_localize(None)
    assert truth_ti.equals(test[2]), 'time_index does not match!'


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
                       check_categorical=False)

    path = os.path.join(ROOT_DIR, 'ReEDS_Timeslice_rep_coeffs.csv')
    truth = pd.read_csv(path)
    assert_frame_equal(truth, test_coeffs, check_dtype=False,
                       check_categorical=False)


def test_cf_timeslices():
    """
    Test ReedsTimeslices from CF profiles
    """
    cf_profiles = os.path.join(TESTDATADIR, 'reV_gen', 'gen_pv_2012.h5')
    rev_table = os.path.join(ROOT_DIR, 'ReEDS_Classifications.csv')
    timeslice_map = os.path.join(ROOT_DIR, 'inputs',
                                 'timeslices_hb.csv')
    test, _ = ReedsTimeslices.run(cf_profiles, timeslice_map,
                                  rev_table=rev_table, max_workers=1,
                                  legacy_format=True)

    path = os.path.join(ROOT_DIR, 'ReEDS_Timeslice_cf_stats.csv')
    truth = pd.read_csv(path)
    assert_frame_equal(truth, test, check_dtype=False,
                       check_categorical=False)


def test_timeslice_h5_output():
    """
    Test h5 output for timeslice correlation table.
    """

    rep_profiles = os.path.join(ROOT_DIR, 'ReEDS_Profiles.h5')
    timeslice_map = os.path.join(ROOT_DIR, 'inputs',
                                 'timeslices.csv')
    reg_cols = ('region', 'class')
    _, test_coeffs = ReedsTimeslices.run(rep_profiles, timeslice_map,
                                         max_workers=1,
                                         legacy_format=False,
                                         reg_cols=reg_cols)

    with tempfile.TemporaryDirectory() as td:
        fpath = os.path.join(td, 'ReEDS_Corr_Coeffs.h5')
        ReedsTimeslices.save_correlation_dict(test_coeffs, reg_cols, fpath,
                                              sparsify=True)

        with Resource(fpath) as out:
            meta = out.meta
            indices = out['indices']
            for k, v in test_coeffs.items():
                dset = 'timeslice_{}'.format(k)
                data = \
                    ReedsTimeslices.unsparsify_corr_matrix(out[dset], indices)
                assert np.allclose(data, np.round(v, decimals=3), atol=0.001)
                assert len(meta) == len(data)


@pytest.fixture
def offshore_trgs():
    """
    load TRG classes
    """
    path = os.path.join(TESTDATADIR, 'reeds/inputs/reeds_class_bins.csv')
    trg_classes = pd.read_csv(path)
    return trg_classes[['class', 'TRG_cap', 'sub_type']]


@pytest.fixture
def offshore_mean_res():
    """
    load TRG classes
    """
    path = os.path.join(TESTDATADIR, 'reeds/inputs/reeds_class_bins.csv')
    trg_classes = pd.read_csv(path)
    return trg_classes[['class', 'mean_res_min', 'mean_res_max', 'sub_type']]


def test_offshore_classifier():
    """
    Test ReedsClassifier with offshore filtering
    """
    rev_table = os.path.join(ROOT_DIR, 'inputs/ri_wind_farm_sc.csv')
    path = os.path.join(TESTDATADIR, 'reeds/inputs/reeds_class_bins.csv')
    resource_classes = pd.read_csv(path)

    # TRG Classes
    path = os.path.join(ROOT_DIR, 'ReEDS_Offshore_TRG_Classifications.csv')
    truth_table = pd.read_csv(path)
    class_bins = resource_classes.copy()
    class_bins = class_bins[['class', 'TRG_cap', 'sub_type']]

    test_table = ReedsClassifier.create(rev_table, class_bins,
                                        region_map='reeds_region',
                                        cap_bins=3, sort_bins_by='mean_lcoe',
                                        pre_filter={'offshore': 1},
                                        trg_by_region=True)[0]
    assert_frame_equal(truth_table, test_table, check_dtype=False,
                       check_categorical=False)

    # Range bins on mean_res
    path = os.path.join(ROOT_DIR, 'ReEDS_Offshore_res_Classifications.csv')
    truth_table = pd.read_csv(path)
    class_bins = resource_classes.copy()
    class_bins = class_bins[[
        'class', 'mean_res_min', 'mean_res_max', 'sub_type']]

    test_table = ReedsClassifier.create(rev_table, class_bins,
                                        region_map='reeds_region',
                                        cap_bins=3, sort_bins_by='mean_lcoe',
                                        pre_filter={'offshore': 1})[0]
    assert_frame_equal(truth_table, test_table, check_dtype=False,
                       check_categorical=False)


def test_wind_farm_profiles():
    """Integrated baseline test for REEDS representative profile calculation
    from SC wind farm aggregate profiles"""

    cf_profiles = os.path.join(ROOT_DIR, 'inputs/ri_wind_farm_profiles.h5')
    rev_table = os.path.join(ROOT_DIR, 'inputs/ri_wind_farm_sc.csv')
    path = os.path.join(TESTDATADIR, 'reeds/inputs/reeds_class_bins.csv')
    resource_classes = pd.read_csv(path)
    resource_classes = resource_classes[['class', 'TRG_cap', 'sub_type']]

    rev_table = ReedsClassifier.create(rev_table, resource_classes,
                                       region_map='reeds_region',
                                       cap_bins=2, sort_bins_by='mean_lcoe',
                                       pre_filter={'offshore': 1})[0]

    f_baseline = os.path.join(ROOT_DIR, 'ReEDS_Wind_Farm_Profiles.h5')

    gid_col = 'sc_gid'
    profiles_dset = 'rep_profiles_0'
    weight = 'capacity'

    profiles, meta, _ = ReedsProfiles.run(cf_profiles, rev_table,
                                          gid_col=gid_col,
                                          profiles_dset=profiles_dset,
                                          weight=weight,
                                          max_workers=1)

    with Resource(cf_profiles) as res:
        raw_profiles = res[profiles_dset]

    for i, row in meta.iterrows():

        mask = None

        for clabel in row.index:
            if clabel not in ('rep_gen_gid', 'rep_res_gid'):
                temp = rev_table[clabel] == row[clabel]
                if mask is None:
                    mask = temp
                else:
                    mask = mask & temp

        sub = rev_table.loc[mask]
        gids = sub[gid_col].values

        assert row['rep_gen_gid'] in gids
        # pylint: disable=unsubscriptable-object
        assert np.allclose(np.roll(profiles[0][:, i], 1 - row['timezone']),
                           raw_profiles[:, row['rep_gen_gid']])

    with Resource(f_baseline) as res:
        baseline = res[profiles_dset]

    assert np.allclose(np.round(profiles[0], decimals=3), baseline)


def test_sparse_matrix():
    """Test matrix sparsification methods."""

    x = np.arange(100).reshape((10, 10)) * 2 + 1
    try:
        ReedsTimeslices.sparsify_corr_matrix(x)
    except ValueError:
        pass
    else:
        raise Exception('Test failed, should have raised a value error')

    for a in [10, 13, 30]:
        n = a ** 2
        x = np.arange(n).reshape((a, a)) * 2 + 1
        for i in range(len(x)):
            for j in range(len(x)):
                x[i, j] = x[j, i]

        out, indices = ReedsTimeslices.sparsify_corr_matrix(x)
        temp = x.flatten()
        for i, j in enumerate(indices):
            assert out[i] == temp[j]
        assert out[-1] == x[-1, -1]
        assert out[-3] == x[-2, -2]
        sym = ReedsTimeslices.unsparsify_corr_matrix(out, indices)
        assert np.array_equal(x, sym)


def test_cli(runner, trg_classes):
    """
    Test CLI
    """
    path = os.path.join(ROOT_DIR, 'ReEDS_Classifications.csv')
    truth_table_full = pd.read_csv(path)
    path = os.path.join(ROOT_DIR, 'ReEDS_Classifications_Slim.csv')
    truth_table = pd.read_csv(path)
    path = os.path.join(ROOT_DIR, 'ReEDS_Aggregation.csv')
    truth_agg = pd.read_csv(path).fillna(0)

    truth_profiles = \
        extract_profiles(os.path.join(ROOT_DIR, 'ReEDS_Profiles.h5'))

    path = os.path.join(ROOT_DIR, 'ReEDS_Timeslice_rep_stats.csv')
    truth_stats = pd.read_csv(path)

    rev_table = os.path.join(TESTDATADIR, 'reV_sc', 'sc_table.csv')
    cf_profiles = os.path.join(TESTDATADIR, 'reV_gen', 'gen_pv_2012.h5')
    timeslice_map = os.path.join(ROOT_DIR, 'inputs', 'timeslices.csv')
    with tempfile.TemporaryDirectory() as td:
        res_classes = os.path.join(td, 'res_classes.csv')
        trg_classes.to_csv(res_classes, index=False)

        config = {
            "directories": {
                "log_directory": td,
                "output_directory": td
            },
            "execution_control": {
                "option": "local"
            },
            "classify": {
                "rev_table": rev_table,
                "resource_classes": res_classes,
                "cap_bins": 5
            },
            "profiles": {
                "reeds_table": None,
                "cf_profiles": cf_profiles,
                "n_profiles": 3,
                "reg_cols": ['region', 'class']
            },
            "timeslices": {
                "profiles": None,
                "timeslices": timeslice_map
            }
        }
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config',
                                      '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        # Check Classificication Tables
        name = os.path.basename(td)
        path = os.path.join(td,
                            '{}_ReEDS_supply_curve_raw_full.csv'.format(name))
        test_table_full = pd.read_csv(path)
        assert_frame_equal(truth_table_full, test_table_full,
                           check_dtype=False, check_categorical=False)
        path = os.path.join(td, '{}_ReEDS_supply_curve_raw.csv'.format(name))
        test_table = pd.read_csv(path)
        assert_frame_equal(truth_table, test_table, check_dtype=False,
                           check_categorical=False)
        path = os.path.join(td, '{}_ReEDS_supply_curve.csv'.format(name))
        test_agg = pd.read_csv(path).fillna(0)
        assert_frame_equal(truth_agg, test_agg, check_dtype=False,
                           check_categorical=False)

        # Check Profiles
        path = os.path.join(td, '{}_ReEDS_hourly_cf.h5'.format(name))
        test_profiles = extract_profiles(path)
        for k, v in truth_profiles[0].items():
            msg = 'Representative profiles {} do not match!'.format(k)
            assert np.allclose(v, test_profiles[0][k]), msg

        assert_frame_equal(truth_profiles[1], test_profiles[1],
                           check_dtype=False,
                           check_categorical=False)

        msg = 'time_index does not match!'
        assert truth_profiles[2].equals(test_profiles[2]), msg

        # Check timeslices
        path = os.path.join(td, '{}_ReEDS_performance.csv'.format(name))
        test_stats = pd.read_csv(path)
        assert_frame_equal(truth_stats, test_stats, check_dtype=False,
                           check_categorical=False)

        path = os.path.join(td, '{}_ReEDS_hourly_cf.h5'.format(name))
        _, truth_coeffs = ReedsTimeslices.run(path, timeslice_map,
                                              max_workers=1,
                                              legacy_format=False,)
        path = os.path.join(td, '{}_ReEDS_correlations.h5'.format(name))
        with Resource(path) as out:
            meta = out.meta
            for k, v in truth_coeffs.items():
                dset = 'timeslice_{}'.format(k)
                data = out[dset]
                assert np.allclose(data, np.round(v, decimals=3), atol=0.001)
                assert len(meta) == len(data)

    LOGGERS.clear()


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
