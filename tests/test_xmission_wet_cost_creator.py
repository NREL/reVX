"""
Test least cost transmission offshore cost creation
"""
import os
import pytest

import numpy as np

from reVX.least_cost_xmission.costs.wet_cost_creator import (BinConfig,
                                                             WetCostCreator)


def test_bin_config_sanity_checking():
    """
    Test cost binning config sanity checking.
    """
    input = np.array([0, 0])

    reverse_bin = BinConfig(min=10, max=0, cost=1)
    with pytest.raises(AttributeError) as _:
        WetCostCreator._assign_values_by_bins(input, reverse_bin)

    good_config = [
        BinConfig(min=1, max=2, cost=3),
        BinConfig(min=2, max=5, cost=4)
    ]
    WetCostCreator._assign_values_by_bins(input, good_config)


def test_cost_binning_results():
    """ Test results of creating cost raster using bins """
    input = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
    config = [
        BinConfig(max=2, cost=1),
        BinConfig(min=2, max=4, cost=2),
        BinConfig(min=4, max=8, cost=3),
        BinConfig(min=8, cost=4)
    ]
    output = WetCostCreator._assign_values_by_bins(input, config)
    assert (output == np.array([[1, 2, 2],
                                [3, 3, 3],
                                [3, 4, 4]])).all()

    config = [
        BinConfig(min=2, max=4, cost=2),
        BinConfig(min=4, max=8, cost=3),
    ]
    output = WetCostCreator._assign_values_by_bins(input, config)
    assert (output == np.array([[0, 2, 2],
                                [3, 3, 3],
                                [3, 0, 0]])).all()

    input = np.array([[-600, -400, -50],
                      [-700, -250, 70],
                      [-500, -150, -70]])
    config = [
        BinConfig(max=-500, cost=999),
        BinConfig(min=-500, max=-300, cost=666),
        BinConfig(min=-300, max=-100, cost=333),
        BinConfig(min=-100, cost=111)
    ]
    output = WetCostCreator._assign_values_by_bins(input, config)
    assert (output == np.array([[999, 666, 111],
                                [999, 333, 111],
                                [666, 333, 111]])).all()


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
