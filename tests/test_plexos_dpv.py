# -*- coding: utf-8 -*-
"""reVX PLEXOS unit test module
"""
import os
import pytest
import pandas as pd
import numpy as np
from reVX import TESTDATADIR
from reV.handlers.resource import Resource
from reVX.plexos.dpv_plexos import DpvResource, DpvPlexosAggregation

ROOT_DIR = os.path.join(TESTDATADIR, 'reV_gen/')


def test_merge():
    """Test dpv tech merge against simple calc"""

    job_frac_map = {'dgen_a90_t28': 0.5,
                    'dgen_a135_t28': 0.5,
                    }
    sub_dirs = list(job_frac_map.keys())

    dpv = DpvResource(ROOT_DIR, sub_dirs, 2007)
    arr = dpv._merge_data('cf_mean', job_frac_map)

    fn1 = os.path.join(ROOT_DIR, 'dgen_a90_t28/dgen_a90_t28_gen_2007.h5')
    fn2 = os.path.join(ROOT_DIR, 'dgen_a135_t28/dgen_a135_t28_gen_2007.h5')
    with Resource(fn1) as res1:
        with Resource(fn2) as res2:
            truth = (res1['cf_mean'] + res2['cf_mean']) / 2

    assert np.allclose(arr, truth, rtol=0.01)


def test_merge_baseline():
    """Test dpv tech merge against baseline file"""

    job_frac_map = {'dgen_a90_t28': 0.13,
                    'dgen_a135_t28': 0.09,
                    'dgen_a180_t28': 0.25,
                    'dgen_a225_t28': 0.09,
                    'dgen_a270_t28': 0.13,
                    'dgen_t0': 0.31,
                    }
    sub_dirs = list(job_frac_map.keys())

    dpv = DpvResource(ROOT_DIR, sub_dirs, 2007)
    arr = dpv._merge_data('cf_mean', job_frac_map)

    with Resource(os.path.join(ROOT_DIR, 'naris_rev_dpv_2007.h5')) as res:
        truth = res['cf_mean']

    assert np.allclose(arr, truth, rtol=0.01)


def test_dpv_agg():
    """Test aggregation of DPV rev results to plexos nodes"""
    cf = os.path.join(ROOT_DIR, 'naris_rev_dpv_2007.h5')
    with Resource(cf) as res:
        meta = res.meta
        truth = res['cf_profile', :, [0, 1]]

    lats = [meta.latitude[0], meta.latitude[1]]
    lons = [meta.longitude[0], meta.longitude[1]]
    cap = [50, 100]
    node_buildout = pd.DataFrame({'plexos_id': [0, 1],
                                  'latitude': lats,
                                  'longitude': lons,
                                  'built_capacity': cap})
    pa = DpvPlexosAggregation(node_buildout, cf)
    profiles = pa.get_node_gen_profiles()
    truth *= cap
    assert np.allclose(profiles, truth)
    return profiles, truth


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
