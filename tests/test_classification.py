# -*- coding: utf-8 -*-
"""reVX PLEXOS unit test module
"""
import os
import pytest
import pandas as pd
from pandas.testing import assert_series_equal

pytest.importorskip("rtree")

from reVX import TESTDATADIR
from reVX.utilities.region import RegionClassifier


META_PATH = os.path.join(TESTDATADIR, 'classification/meta.csv')
REGIONS_PATH = os.path.join(TESTDATADIR, 'classification/us_states.shp')
RESULTS_PATH = os.path.join(TESTDATADIR, 'classification/new_meta.csv')

REGIONS_LABEL = 'NAME'


def test_region_classification():
    """Test the rpm clustering pipeline and run a baseline validation."""

    classification = RegionClassifier.run(meta_path=META_PATH,
                                          regions_path=REGIONS_PATH,
                                          regions_label=REGIONS_LABEL,
                                          force=True)

    test_labels = classification[REGIONS_LABEL]
    valid_labels = pd.read_csv(RESULTS_PATH)[REGIONS_LABEL]
    assert_series_equal(test_labels, valid_labels)


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
