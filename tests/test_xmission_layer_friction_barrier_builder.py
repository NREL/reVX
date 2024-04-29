
"""
Test least cost transmission friction and barrier building
"""
import os
import pytest

import numpy as np

from reVX.handlers.layered_h5 import LayeredTransmissionH5
from reVX.config.transmission_layer_creation import FBLayerConfig, RangeConfig
from reVX.least_cost_xmission.layers.masks import Masks
from reVX.least_cost_xmission.layers.friction_barrier_builder import (
    FrictionBarrierBuilder
)


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

builder = FrictionBarrierBuilder(io_handler, masks)


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


def test_range():
    """ Test range key in FBLayerConfig """
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    config = FBLayerConfig(
        extent='wet+',
        range=[RangeConfig(min=1, max=5, value=4)]
    )
    result = builder._process_raster_layer(data, config)
    assert (result == np.array([[4, 4, 0],
                                [4, 0, 0],
                                [0, 0, 0]])).all()

    config = FBLayerConfig(
        extent='dry+',
        range=[RangeConfig(min=5, max=9, value=5)]
    )
    result = builder._process_raster_layer(data, config)
    assert (result == np.array([[0, 0, 0],
                                [0, 5, 5],
                                [0, 5, 0]])).all()

    config = FBLayerConfig(
        extent='all',
        range=[RangeConfig(min=2, max=9, value=5)]
    )
    result = builder._process_raster_layer(data, config)
    assert (result == np.array([[0, 5, 5],
                                [5, 5, 5],
                                [5, 5, 0]])).all()


def test_complex_ranges():
    """ Test range key with multiple ranges in FBLayerConfig """
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    config = FBLayerConfig(
        extent='wet+',
        range=[
            RangeConfig(min=1, max=5, value=4),
            RangeConfig(min=4, max=10, value=10)
        ]
    )
    result = builder._process_raster_layer(data, config)
    assert (result == np.array([[4, 4, 0],
                                [14, 10, 0],
                                [10, 10, 0]])).all()

    config = FBLayerConfig(
        extent='all',
        range=[
            RangeConfig(min=1, max=6, value=5),
            RangeConfig(min=4, max=9, value=1)
        ]
    )
    result = builder._process_raster_layer(data, config)
    assert (result == np.array([[5, 5, 5],
                                [6, 6, 1],
                                [1, 1, 0]])).all()


def test_map():
    """ Test map key in FBLayerConfig """
    data = np.array([[1, 1, 1],
                     [2, 2, 2],
                     [3, 3, 3]])
    config = FBLayerConfig(
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
    config = FBLayerConfig(
        extent='landfall',
        map={1: 5, 3: 9},
    )
    result = builder._process_raster_layer(data, config)
    assert (result == np.array([[0, 5, 0],
                                [0, 0, 0],
                                [0, 9, 0]])).all()


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
