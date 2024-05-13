
"""
Test least cost transmission friction and barrier building
"""
import os
import pytest
from typing import Tuple

import numpy as np
import numpy.typing as npt

from reVX.handlers.layered_h5 import LayeredTransmissionH5
from reVX.config.transmission_layer_creation import LayerBuildConfig
from reVX.least_cost_xmission.layers import LayerCreator
from reVX.least_cost_xmission.layers.masks import Masks

global_result: npt.NDArray


class FakeIoHandler:
    """ Fake IO Handler for testing """
    def __init__(self, shape: Tuple[int, int]):
        self.shape = shape

    # pylint: disable=unused-argument
    def load_data_using_h5_profile(self, fname: str,
                                   reproject: bool = True) -> npt.NDArray:
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

        raise AttributeError(f"Unknown filename: {fname}")

    # pylint: disable=unused-argument
    def save_data_using_h5_profile(self, data: npt.NDArray, fname: str):
        """ Store data to be saved in GeoTIFF to global """
        global global_result  # pylint: disable=global-statement
        global_result = data

    def write_layer_to_h5(self, data, layer_name):
        pass


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


def test_forced_inclusion():
    """ Test forced inclusions """
    config = {
        'fi_1.tif': LayerBuildConfig(
            extent='wet+',
            forced_inclusion=True,
        ),
        'friction_1.tif': LayerBuildConfig(
            extent='all',
            map={1: 1, 2: 2, 3: 3}
        ),
        'fi_2.tif': LayerBuildConfig(
            extent='dry+',
            forced_inclusion=True,
        ),
    }
    builder = LayerCreator(io_handler, masks)
    builder.build('friction', config, write_to_h5=False)
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
