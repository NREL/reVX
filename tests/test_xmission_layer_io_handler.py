# -*- coding: utf-8 -*-
"""``TransLayerIoHandler`` tests"""
import os
import tempfile

import pytest

from reVX import TESTDATADIR
from reVX.least_cost_xmission.layers.transmission_layer_io_handler import (
    TransLayerIoHandler
)


ISO_REGIONS_FP = os.path.join(TESTDATADIR, 'xmission', 'ri_regions.tif')
XMISSION_LAYERS_H5 = os.path.join(TESTDATADIR, 'xmission',
                                  'xmission_layers.h5')


def test_init_h5():
    """Test initializing an H5 file with `TransLayerIoHandler`"""

    io_handler = TransLayerIoHandler(ISO_REGIONS_FP)
    assert io_handler.h5_file is None

    with tempfile.TemporaryDirectory() as td:
        new_h5_fp = os.path.join(td, "test.h5")
        with pytest.raises(FileNotFoundError):
            io_handler.h5_file = new_h5_fp

        io_handler.create_new_h5(XMISSION_LAYERS_H5, new_h5_fp)
        io_handler.h5_file = new_h5_fp  # no error

        with pytest.raises(FileExistsError):
            io_handler.create_new_h5(XMISSION_LAYERS_H5, new_h5_fp)

        # no error
        io_handler.create_new_h5(XMISSION_LAYERS_H5, new_h5_fp, overwrite=True)


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
