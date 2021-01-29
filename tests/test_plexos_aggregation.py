# -*- coding: utf-8 -*-
"""reVX PLEXOS unit test module
"""
import os
import pytest
import pandas as pd
import numpy as np

from reVX.plexos.rev_reeds_plexos import PlexosAggregation
from reVX import TESTDATADIR

REV_SC = os.path.join(
    TESTDATADIR,
    'reV_sc/wtk_coe_2017_cem_v3_wind_conus_multiyear_colorado.csv')
REEDS = os.path.join(TESTDATADIR, 'plexos/reeds_build.csv')
CF_FPATH = os.path.join(TESTDATADIR,
                        'reV_gen/naris_rev_wtk_gen_colorado_2007.h5')
PLEXOS_NODES = os.path.join(TESTDATADIR, 'plexos/PLEXOS_NODES.csv')


def test_plexos_agg():
    """Test that a plexos node aggregation matches baseline results."""

    outdir = os.path.join(os.path.dirname(__file__),
                          'data/aggregated_plexos_profiles/')

    build_year = 2050
    plexos_meta, time_index, profiles = PlexosAggregation.run(
        PLEXOS_NODES, REV_SC, REEDS, CF_FPATH, build_year=build_year)

    fpath_meta = os.path.join(outdir, 'plexos_meta.csv')
    fpath_profiles = os.path.join(outdir, 'profiles.csv')

    if not os.path.exists(fpath_meta):
        plexos_meta.to_csv(fpath_meta)

    if not os.path.exists(fpath_profiles):
        pd.DataFrame(profiles).to_csv(fpath_profiles)

    baseline_meta = pd.read_csv(fpath_meta, index_col=0)
    baseline_profiles = pd.read_csv(fpath_profiles, index_col=0)

    assert all(baseline_meta['gen_gids'] == plexos_meta['gen_gids'])
    assert np.allclose(baseline_meta['built_capacity'],
                       plexos_meta['built_capacity'])
    assert np.allclose(baseline_profiles.values, profiles)

    return plexos_meta, time_index, profiles


def test_missing_gids():
    """Test that buildouts with missing resource gids are allocated correctly.
    """

    build_year = 2050
    rev_sc = pd.read_csv(REV_SC)
    # add missing SC points to requested buildout for test
    n = 10
    missing_test = pd.DataFrame({'gid': rev_sc.iloc[0:n]['gid'],
                                 'year': [build_year] * n,
                                 'capacity_reV': [10] * n})

    reeds = pd.read_csv(REEDS)
    reeds = reeds.append(missing_test, ignore_index=False)
    icap = reeds['capacity_reV'].sum()

    pa = PlexosAggregation(PLEXOS_NODES, rev_sc, reeds, CF_FPATH)
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
