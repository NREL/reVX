# -*- coding: utf-8 -*-
"""reVX PLEXOS unit test module
"""
import os
import pytest
import pandas as pd
import numpy as np
from reV.handlers.resource import Resource
from reVX.plexos.rev_reeds_plexos import PlexosNode, DataCleaner


@pytest.fixture
def sc_build():
    """Initialize a sc_build dataframe."""
    datadir = os.path.join(os.path.dirname(__file__), 'data/')

    fn = os.path.join(
        datadir,
        'reV_sc/wtk_coe_2017_cem_v3_wind_conus_multiyear_colorado.csv')
    rev_sc = pd.read_csv(fn)

    fn = os.path.join(
        datadir,
        'reeds/BAU_wtk_coe_2017_cem_v3_wind_conus_'
        'multiyear_US_wind_reeds_to_rev.csv')
    reeds = pd.read_csv(fn)

    rev_sc = DataCleaner.rename_cols(rev_sc, DataCleaner.REV_NAME_MAP)
    reeds = DataCleaner.rename_cols(reeds, DataCleaner.REEDS_NAME_MAP)

    year_mask = (reeds.reeds_year == 2050)
    reeds = reeds[year_mask]
    sc_build = pd.merge(rev_sc, reeds, how='inner', on='gid')

    return sc_build


@pytest.fixture
def cf_fpath():
    """Get a reV gen cf test filepath."""
    datadir = os.path.join(os.path.dirname(__file__), 'data/')

    cf_fpath = os.path.join(
        datadir, 'reV_gen/naris_rev_wtk_gen_colorado_2007.h5')

    return cf_fpath


def get_cf_attrs(cf_fpath):
    """Get the time index and available resource gids from a cf file"""
    with Resource(cf_fpath) as res:
        time_index = res.time_index
        cf_res_gids = res.get_meta_arr('gid')

    return time_index, cf_res_gids


def test_plexos_node_build(sc_build, cf_fpath):
    """Test that a plexos node buildout has consistent results."""
    time_index, cf_res_gids = get_cf_attrs(cf_fpath)
    sc_build = sc_build.iloc[0:5]
    x = PlexosNode.run(sc_build, cf_fpath, cf_res_gids,
                       power_density=3, exclusion_area=0.0081)

    profile, res_gids, gen_gids, res_built = x

    assert len(profile) == len(time_index)
    assert len(res_gids) == len(gen_gids)
    assert len(res_gids) == len(res_built)

    err = 100 * np.abs((np.sum(res_built) - np.sum(sc_build.built_capacity))
                       / np.sum(res_built))
    assert err < 1, 'Built capacity does not match desired to within 1%'

    return x


def test_plexos_node_profile(sc_build, cf_fpath):
    """Test that a plexos node buildout profile matches source data"""
    _, cf_res_gids = get_cf_attrs(cf_fpath)
    sc_build = sc_build.iloc[100:102]
    x = PlexosNode.run(sc_build, cf_fpath, cf_res_gids,
                       power_density=3, exclusion_area=0.0081)

    profile, _, gen_gids, res_built = x

    with Resource(cf_fpath) as res:
        arr = res['cf_profile', :, gen_gids]

    arr *= res_built
    arr = arr.sum(axis=1)

    diff = np.abs(profile - arr) / profile
    diff = np.nan_to_num(diff)

    # small difference is attributed to rounding error in res_built values
    assert diff.max() < 0.01, 'Buildout profiles are inconsistent'
    assert diff.mean() < 0.005, 'Buildout profiles are inconsistent'

    return x


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
