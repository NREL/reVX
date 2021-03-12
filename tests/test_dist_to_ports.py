# -*- coding: utf-8 -*-
"""
Distance to Ports tests
"""
from click.testing import CliRunner
import json
import numpy as np
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import shutil
import tempfile
import traceback

from rex.resource import Resource
from rex.utilities.loggers import LOGGERS
from reV.handlers.exclusions import ExclusionLayers
from reVX import TESTDATADIR
from reVX.offshore.dist_to_ports import DistanceToPorts, AssemblyAreas
from reVX.offshore.dist_to_ports_cli import main

EXCL_H5 = os.path.join(TESTDATADIR, 'offshore', 'offshore.h5')
PORTS_FPATH = os.path.join(TESTDATADIR, 'offshore', 'ports',
                           'ports_operations.shp')
ASSEMBLY_AREAS = os.path.join(TESTDATADIR, 'offshore', 'assembly_areas.csv')


def get_dist_to_ports(excl_h5, ports_layer='ports_operations'):
    """
    Extract "truth" distance to ports layer from exclusion .h5 file
    """
    with ExclusionLayers(excl_h5) as f:
        dist_to_ports = f[ports_layer]

    return dist_to_ports


def get_assembly_areas(excl_h5, assembly_dset='assembly_areas'):
    """
    Extract "truth" assembly areas table
    """
    with Resource(excl_h5) as f:
        assembly_areas = f.df_str_decode(pd.DataFrame(f[assembly_dset]))

    return assembly_areas


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


@pytest.mark.parametrize('max_workers', [None, 1])
def test_dist_to_ports(max_workers):
    """
    Compute distance to ports
    """
    baseline = get_dist_to_ports(EXCL_H5)
    test = DistanceToPorts.run(PORTS_FPATH, EXCL_H5, max_workers=max_workers)

    msg = 'distance to ports does not match baseline distances'
    assert np.allclose(baseline, test), msg


def test_assembly_area():
    """
    Compute distance from ports to assembly areas
    """
    truth = get_assembly_areas(EXCL_H5)
    test = AssemblyAreas.run(ASSEMBLY_AREAS, EXCL_H5)

    assert_frame_equal(truth, test, check_dtype=False)


@pytest.mark.parametrize('ports_layer', [None, 'test'])
def test_cli(runner, ports_layer):
    """
    Test CLI
    """
    update = False
    if ports_layer is None:
        update = True
        ports_layer = 'ports_operations'

    with tempfile.TemporaryDirectory() as td:
        excl_fpath = os.path.basename(EXCL_H5)
        excl_fpath = os.path.join(td, excl_fpath)
        shutil.copy(EXCL_H5, excl_fpath)
        config = {
            "directories": {
                "log_directory": td,
            },
            "execution_control": {
                "option": "local"
            },
            "excl_fpath": excl_fpath,
            "ports_fpath": PORTS_FPATH,
            "output_dist_layer": ports_layer,
            "update": update,
            "assembly_areas": ASSEMBLY_AREAS
        }
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config',
                                      '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        baseline = get_dist_to_ports(EXCL_H5)
        test = get_dist_to_ports(excl_fpath, ports_layer=ports_layer)

        msg = 'distance to ports does not match baseline distances'
        assert np.allclose(baseline, test), msg

        truth = get_assembly_areas(EXCL_H5)
        test = get_assembly_areas(excl_fpath)
        assert_frame_equal(truth, test, check_dtype=False)

    LOGGERS.clear()


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
