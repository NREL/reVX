
"""
Test least cost transmission friction and barrier building
"""
import os
import pytest

import numpy as np
from reVX.least_cost_xmission.friction_barrier_builder import FrictionBarrierBuilder

from reVX.least_cost_xmission.masks import Masks

# Fake masks. Left side of array is wet, right side is dry, center column in
# landfall
masks = Masks('fake_handler')
masks._dry_mask = np.array([[False, False, True],
                            [False, False, True],
                            [False, False, True]])

masks._wet_mask = np.array([[True, False, False],
                            [True, False, False],
                            [True, False, False]])

masks._landfall_mask = np.array([[False, True, False],
                                 [False, True, False],
                                 [False, True, False]])

class FakeIoHandler:
    def __init__(self, shape):
        self.shape = shape

io_handler = FakeIoHandler((3, 3))
builder = FrictionBarrierBuilder('friction', io_handler, masks)


def test_mask_plus():
    """ Test function of wet_plus and dry_plus masks """
    assert (
        masks.dry_plus_mask ==
        np.array([[False, True, True],
                  [False, True, True],
                  [False, True, True]])
    ).all()

    assert (
        masks.wet_plus_mask ==
        np.array([[True, True, False],
                  [True, True, False],
                  [True, True, False]])
    ).all()


def test_range():
    """ Test range key in LayerConfig """
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    config = {
        'extent': 'wet+',
        'range': [1, 5],
        'value': 4
    }
    result = builder._process_layer(data, config)
    assert (result == np.array([[4, 4, 0],
                                [4, 0, 0],
                                [0, 0, 0]])).all()

    config = {
        'extent': 'dry+',
        'range': [5, 9],
        'value': 5
    }
    result = builder._process_layer(data, config)
    assert (result == np.array([[0, 0, 0],
                                [0, 5, 5],
                                [0, 5, 0]])).all()

    config = {
        'extent': 'all',
        'range': [2, 9],
        'value': 5
    }
    result = builder._process_layer(data, config)
    assert (result == np.array([[0, 5, 5],
                                [5, 5, 5],
                                [5, 5, 0]])).all()


def test_map():
    """ Test map key in LayerConfig """
    data = np.array([[1, 1, 1],
                     [2, 2, 2],
                     [3, 3, 3]])
    config = {
        'extent': 'wet',
        'map': {1: 5, 3: 9},
    }
    result = builder._process_layer(data, config)
    assert (result == np.array([[5, 0, 0],
                                [0, 0, 0],
                                [9, 0, 0]])).all()

    data = np.array([[1, 1, 1],
                     [2, 2, 2],
                     [3, 3, 3]])
    config = {
        'extent': 'landfall',
        'map': {1: 5, 3: 9},
    }
    result = builder._process_layer(data, config)
    assert (result == np.array([[0, 5, 0],
                                [0, 0, 0],
                                [0, 9, 0]])).all()

def test_combine_layers():
    """ Processing and combination of layers """
    layers = [
        (
            np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
            {
                'extent': 'wet',
                'map': {1: 5, 3: 9},
            }
        )
    ]
    result = builder._combine_layers(layers)
    assert (result == np.array([[5, 0, 0],
                                [0, 0, 0],
                                [9, 0, 0]])).all()

    layers = [
        (
            np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
            {
                'extent': 'all',
                'range': [0, 10],
                'value': 100,
            }
        ),
        (
            np.array([[1, 1, 1],
                      [2, 2, 2],
                      [3, 3, 3]]),
            {
                'extent': 'wet+',
                'map': {1: 10, 2: 20},
            }
        ),
        (
            np.array([[10, 10, 10],
                      [20, 20, 20],
                      [30, 30, 30]]),
            {
                'extent': 'dry+',
                'map': {20: 2, 30: 3},
            }
        ),
    ]
    result = builder._combine_layers(layers)
    print(result)
    assert (result == np.array([[110, 110, 100],
                                [120, 122, 102],
                                [100, 103, 103]])).all()

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
