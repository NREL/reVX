# -*- coding: utf-8 -*-
"""
reVX RED-E unit test module
"""
import numpy as np
import os
import pytest

from reVX.red_e.tech_potential import TechPotential
from reVX import TESTDATADIR

EXCL = os.path.join(TESTDATADIR, 'red_e/Brunei_subset.h5')
EXCL_DICT = {'landuse': {'exclude_values': [5],
                         'exclude_nodata': True},
             'protected_area': {'exclude_values': [1],
                                'exclude_nodata': False},
             'adm_boundary_1': {'include_values': [1, 2, 4],
                                'exclude_nodata': True},
             'ghi': {'inclusion_range': (4, None),
                     'exclude_nodata': True},
             'srtm_slope': {'inclusion_range': (None, 5),
                            'exclude_nodata': True},
             'road_distance': {'inclusion_range': (None, 15000),
                               'exclude_nodata': True}}
POWER_DENSITY = 36
RTOL = 0.001


@pytest.mark.parametrize(('base', 'power_density'),
                         (('solar_fixlat_cf', POWER_DENSITY),
                          ('solar_fixlat_cf', 1),
                          ('wind_200_sp_cf', POWER_DENSITY),
                          ('wind_237_sp_cf', POWER_DENSITY),
                          ('wind_245_sp_cf', POWER_DENSITY),
                          ('wind_255_sp_cf', POWER_DENSITY),
                          ('dni', 1)))
def test_tech_pot(base, power_density):
    """
    Test Tech potential
    """
    path = os.path.join(TESTDATADIR, 'red_e',
                        '{}-{}.npy'.format(base, power_density))
    truth = np.load(path)

    if 'cf' in base:
        test = TechPotential.run_generation(EXCL, base, EXCL_DICT,
                                            power_density=power_density)
    else:
        test = TechPotential.run(EXCL, base, EXCL_DICT,
                                 power_density=power_density)

    assert np.allclose(test, truth, rtol=RTOL)
