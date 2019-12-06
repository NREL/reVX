# -*- coding: utf-8 -*-
"""reVX PLEXOS unit test module
"""
import os
import pytest
import pandas as pd
from pandas.testing import assert_series_equal

from reVX import TESTDATADIR as reVX_TESTDATADIR
from reVX.classification.region import region_classifier


META_PATH = os.path.join(reVX_TESTDATADIR, 'classification/meta.csv')
REGIONS_PATH = os.path.join(reVX_TESTDATADIR, 'classification/us_states.shp')
RESULTS_PATH = os.path.join(reVX_TESTDATADIR, 'classification/new_meta.csv')

REGIONS_LABEL = 'NAME'
LAT_LABEL = 'LATITUDE'
LONG_LABEL = 'LONGITUDE'


def test_region_classification():
    """Test the rpm clustering pipeline and run a baseline validation."""

    classifier = region_classifier(meta_path=META_PATH,
                                   regions_path=REGIONS_PATH,
                                   lat_label=LAT_LABEL,
                                   long_label=LONG_LABEL,
                                   regions_label=REGIONS_LABEL)

    classification = classifier.classify(force=True)

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
