
"""
Test least cost transmission friction and barrier building
"""
import os
import pytest
from typing import Tuple

import numpy as np
import numpy.typing as npt

from reVX.least_cost_xmission.layers.masks import Masks
from reVX.least_cost_xmission.layers.friction_barrier_builder import (
    FBLayerConfig, FrictionBarrierBuilder
)
from reVX.least_cost_xmission.layers.transmission_layer_io_handler import (
    TransLayerIoHandler
)

global_result: npt.NDArray


class FakeIoHandler:
    """ Fake IO Handler for testing """
    def __init__(self, shape: Tuple[int, int]):
        self.shape = shape

    def load_tiff(self, fname: str,
                  reproject: bool = True  # pylint: disable=unused-argument
                  ) -> npt.NDArray:
        """ Fake tiff loader """
        if fname == 'friction_1.tif':
            return np.array([[1, 1, 1],
                             [2, 2, 2],
                             [3, 3, 3]])

        if fname == 'fi_1.tif':
            return np.array([[0, 0, 0],
                             [1, 1, 1],
                             [0, 0, 0]])

        if fname == 'fi_2.tif':
            return np.array([[0, 0, 0],
                             [0, 0, 0],
                             [2, 2, 2]])

        raise AttributeError

    def save_tiff(self, data: npt.NDArray,
                  fname: str):  # pylint: disable=unused-argument
        """ Store data to be saved in GeoTIFF to global """
        global global_result  # pylint: disable=global-statement
        global_result = data


io_handler: TransLayerIoHandler = FakeIoHandler((3, 3))  # type: ignore

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


def test_forced_inclusion():
    """ Test forced inclusions """
    config = {
        'fi_1.tif': FBLayerConfig(
            extent='wet+',
            forced_inclusion=True,
        ),
        'friction_1.tif': FBLayerConfig(
            extent='all',
            map={1: 1, 2: 2, 3: 3}
        ),
        'fi_2.tif': FBLayerConfig(
            extent='dry+',
            forced_inclusion=True,
        ),
    }
    builder = FrictionBarrierBuilder('friction', io_handler, masks)
    builder.build_layer(config)
    assert (global_result == np.array([[1, 1, 1],
                                       [0, 0, 2],
                                       [3, 0, 0]])).all()


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
