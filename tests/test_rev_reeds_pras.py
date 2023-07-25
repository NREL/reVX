# -*- coding: utf-8 -*-
"""reVX PLEXOS unit test module
"""
import numpy as np
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import tempfile
import h5py

from rex import Resource, init_logger

from reVX.pras.rev_reeds_pras import PrasAggregation
from reVX import TESTDATADIR

REV_SC = os.path.join(TESTDATADIR, 'reV_sc/sc_upv.csv')
SC_BUILD = os.path.join(TESTDATADIR, 'pras/sc_build.csv')
CF_FPATH = os.path.join(TESTDATADIR, 'reV_gen/gen_upv_2008.h5')


def df_to_sarray(df):
    """
    Convert a pandas DataFrame object to a numpy structured array.
    This is functionally equivalent to but more efficient than
    np.array(df.to_array())

    Parameters
    ----------
    df: the data frame to convert

    Returns
    ------
    a numpy structured array representation of df
    """
    v = df.values
    cols = df.columns
    types = [(k, '<S64') for k in cols]
    z = np.zeros(v.shape[0], types)
    for (i, k) in enumerate(z.dtype.names):
        z[k] = v[:, i]
    return z


def create_pras_test_file(outdir):
    """Create pras test file to use with PrasAggregation"""
    outfile = f'{outdir}/pras_file.pras.h5'
    with h5py.File(outfile, 'w') as out:
        g1 = out.create_group('generators')
        names = []
        categories = []
        regions = []
        for i in range(5):
            for j in (2, 3):
                names.append(f'upv_p{i + 1}_{j}')
                categories.append(f'upv_{j}')
                regions.append(f'p{i + 1}')
        df = pd.DataFrame({'name': names, 'category': categories,
                           'region': regions})
        g1.create_dataset('_core', data=df_to_sarray(df))
        g1.create_dataset('capacity',
                          data=np.zeros((8760, 10), dtype='float32'))
    return outfile


def test_pras_agg_indexing():
    """Test that pras aggregation create correct indices for profiles and pras
    output"""
    resource_class = 3
    with tempfile.TemporaryDirectory() as tmpdir:
        pras_file = create_pras_test_file(tmpdir)
        build_year = 2050
        pa = PrasAggregation(
            REV_SC, SC_BUILD, CF_FPATH, pras_file,
            build_year=build_year, tech_type='upv', res_class=resource_class,
            max_workers=1)
        found_pras_zones = pa.pras_meta['region'][pa.pras_indices].values
        assert np.array_equal(found_pras_zones,
                              pa.sc_build_zones[pa.sc_build_indices])
        with Resource(pras_file) as pras_res:
            df = pd.DataFrame(pras_res['generators/_core'])
            for col in df.columns:
                df[col] = df[col].apply(lambda x: x.decode('utf-8'))
            assert_frame_equal(pa.pras_meta,
                               df[df['category'] == f'upv_{resource_class}'])
            for i, pi in enumerate(pa.pras_indices):
                zone = pa.pras_meta['region'].loc[pi]
                assert df['region'].loc[pi] == zone
                assert pa.sc_build_zones[pa.sc_build_indices[i]] == zone


def test_pras_agg_output(log=True):
    """Test that pras aggregation write profiles to pras output file correctly
    """
    if log:
        init_logger(__name__, log_level='DEBUG')
        init_logger('reVX', log_level='DEBUG')

    with tempfile.TemporaryDirectory() as tmpdir:
        pras_file = create_pras_test_file(tmpdir)
        build_year = 2050
        pa = PrasAggregation(
            REV_SC, SC_BUILD, CF_FPATH, pras_file,
            build_year=build_year, tech_type='upv', res_class=3,
            max_workers=1, timezone='UTC')
        profiles = pa.make_profiles()
        pa.export(pa.time_index, profiles)
        with Resource(pa._output_file) as pras_res:
            for i, pi in enumerate(pa.pras_indices):
                pras_saved = pras_res['generators/capacity'][:, pi]
                prof_out = profiles[:, pa.sc_build_indices[i]]
                assert np.array_equal(pras_saved, prof_out)
                built_cap = pa.built_capacity[i]
                assert np.max(pras_saved) <= built_cap


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
