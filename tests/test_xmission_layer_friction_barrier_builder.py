
"""
Test least cost transmission friction and barrier building
"""
import os
import pytest

import numpy as np

from reVX.handlers.layered_h5 import LayeredTransmissionH5
from reVX.config.transmission_layer_creation import (LayerBuildConfig,
                                                     RangeConfig)
from reVX.least_cost_xmission.layers import LayerCreator
from reVX.least_cost_xmission.layers.masks import Masks


class FakeIoHandler:
    """ Fake IO Handler for testing """
    def __init__(self, shape):
        self.shape = shape


io_handler: LayeredTransmissionH5 = FakeIoHandler((3, 3))  # type: ignore

# Fake masks. Left side of array is wet, right side is dry, center column in
# landfall
masks: Masks = Masks(io_handler)
masks._dry_mask = np.array([[False, False, True],
                            [False, False, True],
                            [False, False, True]])

masks._wet_mask = np.array([[True, False, False],
                            [True, False, False],
                            [True, False, False]])

masks._landfall_mask = np.array([[False, True, False],
                                 [False, True, False],
                                 [False, True, False]])

builder = LayerCreator(io_handler, masks)


def test_mask_plus():
    """ Test function of wet_plus and dry_plus masks """
    assert (
        masks.dry_plus_mask == np.array([[False, True, True],
                                         [False, True, True],
                                         [False, True, True]])
    ).all()

    assert (
        masks.wet_plus_mask == np.array([[True, True, False],
                                         [True, True, False],
                                         [True, True, False]])
    ).all()


def test_bins():
    """ Test bins key in LayerBuildConfig """
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    config = LayerBuildConfig(
        extent='wet+',
        bins=[RangeConfig(min=1, max=5, value=4)]
    )
    result = builder._process_raster_layer(data, config)
    assert (result == np.array([[4, 4, 0],
                                [4, 0, 0],
                                [0, 0, 0]])).all()

    config = LayerBuildConfig(
        extent='dry+',
        bins=[RangeConfig(min=5, max=9, value=5)]
    )
    result = builder._process_raster_layer(data, config)
    assert (result == np.array([[0, 0, 0],
                                [0, 5, 5],
                                [0, 5, 0]])).all()

    config = LayerBuildConfig(
        extent='all',
        bins=[RangeConfig(min=2, max=9, value=5)]
    )
    result = builder._process_raster_layer(data, config)
    assert (result == np.array([[0, 5, 5],
                                [5, 5, 5],
                                [5, 5, 0]])).all()


def test_complex_bins():
    """ Test bins key with multiple bins in LayerBuildConfig """
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    config = LayerBuildConfig(
        extent='wet+',
        bins=[
            RangeConfig(min=1, max=5, value=4),
            RangeConfig(min=4, max=10, value=10)
        ]
    )
    result = builder._process_raster_layer(data, config)
    assert (result == np.array([[4, 4, 0],
                                [14, 10, 0],
                                [10, 10, 0]])).all()

    config = LayerBuildConfig(
        extent='all',
        bins=[
            RangeConfig(min=1, max=6, value=5),
            RangeConfig(min=4, max=9, value=1)
        ]
    )
    result = builder._process_raster_layer(data, config)
    assert (result == np.array([[5, 5, 5],
                                [6, 6, 1],
                                [1, 1, 0]])).all()


def test_map():
    """ Test map key in LayerBuildConfig """
    data = np.array([[1, 1, 1],
                     [2, 2, 2],
                     [3, 3, 3]])
    config = LayerBuildConfig(
        extent='wet',
        map={1: 5, 3: 9},
    )
    result = builder._process_raster_layer(data, config)
    assert (result == np.array([[5, 0, 0],
                                [0, 0, 0],
                                [9, 0, 0]])).all()

    data = np.array([[1, 1, 1],
                     [2, 2, 2],
                     [3, 3, 3]])
    config = LayerBuildConfig(
        extent='landfall',
        map={1: 5, 3: 9},
    )
    result = builder._process_raster_layer(data, config)
    assert (result == np.array([[0, 5, 0],
                                [0, 0, 0],
                                [0, 9, 0]])).all()


def test_bin_config_sanity_checking():
    """
    Test cost binning config sanity checking.
    """
    input = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]])

    reverse_bins = [RangeConfig(min=10, max=0, value=1)]
    config = LayerBuildConfig(extent="all", bins=reverse_bins)
    with pytest.raises(AttributeError) as _:
        builder._process_raster_layer(input, config)

    bin_config = [
        RangeConfig(min=1, max=2, value=3),
        RangeConfig(min=2, max=5, value=4)
    ]
    good_config = LayerBuildConfig(extent="all", bins=bin_config)
    builder._process_raster_layer(input, good_config)


def test_cost_binning_results():
    """ Test results of creating cost raster using bins """
    input = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
    bins = [
        RangeConfig(max=2, value=1),
        RangeConfig(min=2, max=4, value=2),
        RangeConfig(min=4, max=8, value=3),
        RangeConfig(min=8, value=4)
    ]
    config = LayerBuildConfig(extent="all", bins=bins)
    output = builder._process_raster_layer(input, config)
    assert (output == np.array([[1, 2, 2],
                                [3, 3, 3],
                                [3, 4, 4]])).all()

    bins = [
        RangeConfig(min=2, max=4, value=2),
        RangeConfig(min=4, max=8, value=3),
    ]
    config = LayerBuildConfig(extent="all", bins=bins)
    output = builder._process_raster_layer(input, config)
    assert (output == np.array([[0, 2, 2],
                                [3, 3, 3],
                                [3, 0, 0]])).all()

    input = np.array([[-600, -400, -50],
                      [-700, -250, 70],
                      [-500, -150, -70]])
    bins = [
        RangeConfig(max=-500, value=999),
        RangeConfig(min=-500, max=-300, value=666),
        RangeConfig(min=-300, max=-100, value=333),
        RangeConfig(min=-100, value=111)
    ]
    config = LayerBuildConfig(extent="all", bins=bins)
    output = builder._process_raster_layer(input, config)
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
