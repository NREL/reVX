"""
Test least cost transmission offshore cost creation
"""
import os
import pytest

import numpy as np

from reVX.least_cost_xmission.layers.masks import Masks
from reVX.handlers.layered_h5 import LayeredTransmissionH5
from reVX.config.transmission_layer_creation import RangeConfig
from reVX.least_cost_xmission.costs.wet_cost_creator import WetCostCreator


class FakeIoHandler:
    """ Fake IO Handler for testing """
    def __init__(self, shape):
        self.shape = shape


io_handler: LayeredTransmissionH5 = FakeIoHandler((3, 3))  # type: ignore
masks: Masks = Masks(io_handler)
builder = WetCostCreator(io_handler, masks)


def test_bin_config_sanity_checking():
    """
    Test cost binning config sanity checking.
    """
    input = np.array([0, 0])

    reverse_bin = RangeConfig(min=10, max=0, value=1)
    with pytest.raises(AttributeError) as _:
        builder._assign_values_by_bins(input, reverse_bin)

    good_config = [
        RangeConfig(min=1, max=2, value=3),
        RangeConfig(min=2, max=5, value=4)
    ]
    builder._assign_values_by_bins(input, good_config)


def test_cost_binning_results():
    """ Test results of creating cost raster using bins """
    input = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
    config = [
        RangeConfig(max=2, value=1),
        RangeConfig(min=2, max=4, value=2),
        RangeConfig(min=4, max=8, value=3),
        RangeConfig(min=8, value=4)
    ]
    output = builder._assign_values_by_bins(input, config)
    assert (output == np.array([[1, 2, 2],
                                [3, 3, 3],
                                [3, 4, 4]])).all()

    config = [
        RangeConfig(min=2, max=4, value=2),
        RangeConfig(min=4, max=8, value=3),
    ]
    output = builder._assign_values_by_bins(input, config)
    assert (output == np.array([[0, 2, 2],
                                [3, 3, 3],
                                [3, 0, 0]])).all()

    input = np.array([[-600, -400, -50],
                      [-700, -250, 70],
                      [-500, -150, -70]])
    config = [
        RangeConfig(max=-500, value=999),
        RangeConfig(min=-500, max=-300, value=666),
        RangeConfig(min=-300, max=-100, value=333),
        RangeConfig(min=-100, value=111)
    ]
    output = builder._assign_values_by_bins(input, config)
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
