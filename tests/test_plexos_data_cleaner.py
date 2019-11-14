# -*- coding: utf-8 -*-
"""reVX PLEXOS unit test module
"""
import os
import pytest
import pandas as pd
import numpy as np
import json
from reVX.plexos.rev_reeds_plexos import DataCleaner


@pytest.fixture
def rev_sc():
    """Initialize a rev supply curve dataframe."""
    datadir = os.path.join(os.path.dirname(__file__), 'data/')

    fn = os.path.join(
        datadir,
        'reV_sc/wtk_coe_2017_cem_v3_wind_conus_multiyear_colorado.csv')
    rev_sc = pd.read_csv(fn)
    return rev_sc


@pytest.fixture
def plexos_nodes():
    """Initialize a PLEXOS nodes meta dataframe."""
    datadir = os.path.join(os.path.dirname(__file__), 'data/')

    plx_node_fpath = os.path.join(
        datadir, 'plexos/plexos_nodes.csv')
    plexos_nodes = pd.read_csv(plx_node_fpath)

    return plexos_nodes


def buildout_params():
    """Get artificial buildout parameters for testing"""
    built_caps = [19.9, 500, 400]
    gids = ['[0]', '[1]', '[2]']
    gid_builds = ['[4.9]', '[500]', '[400]']
    return built_caps, gids, gid_builds


@pytest.fixture
def plexos_buildout(plexos_nodes):
    """Get a three node plexos buildout meta data"""
    built_caps, gids, gid_builds = buildout_params()
    plexos_buildout = plexos_nodes
    plexos_buildout = plexos_buildout.iloc[0:3]

    plexos_buildout['built_capacity'] = built_caps
    plexos_buildout['res_gids'] = gids
    plexos_buildout['gen_gids'] = gids
    plexos_buildout['res_built'] = gid_builds

    return plexos_buildout


@pytest.fixture
def cf_fpath():
    """Initialize a reV gen cf test fpath."""
    datadir = os.path.join(os.path.dirname(__file__), 'data/')
    cf_fpath = os.path.join(
        datadir, 'reV_gen/naris_rev_wtk_gen_colorado_2007.h5')

    return cf_fpath


def test_df_rename(rev_sc):
    """Test the dataframe column renaming method"""
    rev_sc = DataCleaner.rename_cols(rev_sc, DataCleaner.REV_NAME_MAP)
    for k, v in DataCleaner.REV_NAME_MAP.items():
        assert k not in rev_sc
        assert v in rev_sc


def test_df_reduce(plexos_nodes):
    """Test the method to reduce the columns in a df"""
    plexos_nodes = DataCleaner.reduce_df(plexos_nodes,
                                         DataCleaner.PLEXOS_META_COLS)
    assert 'the_geom_4326' not in plexos_nodes
    assert 'remove' not in plexos_nodes


def test_plexos_pre_filter(plexos_nodes):
    """Test pre-filtering operations of the plexos node meta data"""
    plexos_nodes = DataCleaner.reduce_df(plexos_nodes,
                                         DataCleaner.PLEXOS_META_COLS)
    plexos_nodes = DataCleaner.pre_filter_plexos_meta(plexos_nodes)

    labels = ['latitude', 'longitude']
    a1 = np.sort(np.unique(plexos_nodes[labels].values, axis=0), axis=0)
    a2 = np.sort(plexos_nodes[labels].values, axis=0)

    assert np.allclose(a1, a2)
    assert not any(plexos_nodes.plexos_id.unique() == '#NAME?')


def test_merge_small_nodes(plexos_buildout):
    """Test the merging of small buildout nodes into bigger buildouts."""
    built_caps, gids, gid_builds = buildout_params()

    # force node 0 to build into node 2 by NN
    plexos_buildout.at[2, 'latitude'] = plexos_buildout.at[0, 'latitude']
    plexos_buildout.at[2, 'longitude'] = plexos_buildout.at[0, 'longitude']

    profiles = np.arange(300).reshape((100, 3))

    dc = DataCleaner(plexos_buildout, profiles)
    new_meta, new_profiles = dc.merge_small()

    check1 = (new_meta.loc[2, 'built_capacity']
              == built_caps[2] + built_caps[0])
    check2 = (new_meta.loc[2, 'res_gids']
              == str(json.loads(gids[2]) + json.loads(gids[0])))
    check3 = (new_meta.loc[2, 'res_built']
              == str(json.loads(gid_builds[2]) + json.loads(gid_builds[0])))
    check4 = np.allclose((profiles[:, 0] + profiles[:, 2]), new_profiles[:, 1])

    assert check1
    assert check2
    assert check3
    assert check4


def test_merge_extent(plexos_buildout):
    """Test that two extents can be merged successfully"""
    _, gids, _ = buildout_params()

    profiles = np.arange(300).reshape((100, 3))

    dc = DataCleaner(plexos_buildout.copy(), profiles.copy())
    dc.merge_extent(plexos_buildout, profiles)
    new_meta = dc._plexos_meta
    new_profiles = dc._profiles

    true_gids = [str(2 * json.loads(g)) for g in gids]

    assert np.allclose(new_meta.built_capacity.values,
                       2 * plexos_buildout.built_capacity.values)
    assert np.allclose(new_profiles, 2 * profiles)
    assert true_gids == list(new_meta.res_gids.values)


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
