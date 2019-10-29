# -*- coding: utf-8 -*-
"""reVX PLEXOS unit test module
"""
import os
import pytest
import pandas as pd
import numpy as np
from reVX.plexos.rev_reeds_plexos import PlexosAggregation, DataCleaner


@pytest.fixture
def rev_sc():
    """Initialize a sc_build dataframe."""
    datadir = os.path.join(os.path.dirname(__file__), 'data/')

    fn = os.path.join(
        datadir,
        'reV_sc/wtk_coe_2017_cem_v3_wind_conus_multiyear_colorado.csv')
    rev_sc = pd.read_csv(fn)

    rev_sc = DataCleaner.rename_cols(rev_sc, DataCleaner.REV_NAME_MAP)

    return rev_sc


@pytest.fixture
def reeds():
    """Initialize a sc_build dataframe."""
    datadir = os.path.join(os.path.dirname(__file__), 'data/')

    fn = os.path.join(
        datadir,
        'reeds/BAU_wtk_coe_2017_cem_v3_wind_conus_'
        'multiyear_US_wind_reeds_to_rev.csv')
    reeds = pd.read_csv(fn)

    reeds = DataCleaner.rename_cols(reeds, DataCleaner.REEDS_NAME_MAP)

    return reeds


@pytest.fixture
def cf_fpath():
    """Get a reV gen cf test filepath."""
    datadir = os.path.join(os.path.dirname(__file__), 'data/')

    cf_fpath = os.path.join(
        datadir, 'reV_gen/naris_rev_wtk_gen_colorado_2007.h5')

    return cf_fpath


@pytest.fixture
def plexos_nodes():
    """Initialize a PLEXOS nodes meta dataframe."""
    datadir = os.path.join(os.path.dirname(__file__), 'data/')

    plx_node_fpath = os.path.join(
        datadir, 'plexos/plexos_nodes.csv')
    plexos_nodes = pd.read_csv(plx_node_fpath)

    return plexos_nodes


def test_plexos_agg(plexos_nodes, rev_sc, reeds, cf_fpath):
    """Test that a plexos node aggregation matches baseline results."""

    outdir = os.path.join(os.path.dirname(__file__),
                          'data/aggregated_plexos_profiles/')

    build_year = 2050
    reeds = reeds[reeds.reeds_year == build_year].iloc[500:510]
    plexos_meta, time_index, profiles = PlexosAggregation.run(
        plexos_nodes, rev_sc, reeds, cf_fpath, build_year=build_year)

    fpath_meta = os.path.join(outdir, 'plexos_meta.csv')
    fpath_profiles = os.path.join(outdir, 'profiles.csv')

    if not os.path.exists(fpath_meta):
        plexos_meta.to_csv(fpath_meta)
    if not os.path.exists(fpath_profiles):
        pd.DataFrame(profiles).to_csv(fpath_profiles)

    baseline_meta = pd.read_csv(fpath_meta, index_col=0)
    baseline_profiles = pd.read_csv(fpath_profiles, index_col=0)

    assert all(baseline_meta.gen_gids == plexos_meta.gen_gids)
    assert np.allclose(baseline_meta.built_capacity,
                       plexos_meta.built_capacity)
    assert np.allclose(baseline_profiles.values, profiles)

    return plexos_meta, time_index, profiles


def test_missing_gids(plexos_nodes, rev_sc, reeds, cf_fpath):
    """Test that buildouts with missing resource gids are allocated correctly.
    """

    build_year = 2050
    reeds = reeds[reeds.reeds_year == build_year].iloc[500:510]

    # add missing SC points to requested buildout for test
    n = 10
    missing_test = pd.DataFrame({'gid': rev_sc.iloc[0:n]['gid'],
                                 'reeds_year': [build_year] * n,
                                 'built_capacity': [10] * n})

    reeds = reeds.append(missing_test, ignore_index=False)
    icap = reeds['built_capacity'].sum()

    pa = PlexosAggregation(plexos_nodes, rev_sc, reeds, cf_fpath)
    new_cap = pa._sc_build['built_capacity'].sum()

    assert icap == new_cap

    profiles = pa._make_profiles()

    assert profiles.max(axis=0).sum() / icap > 0.9


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
