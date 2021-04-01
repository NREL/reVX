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
from rex.utilities.utilities import get_lat_lon_cols
from reV.handlers.exclusions import ExclusionLayers
from reVX import TESTDATADIR
from reVX.offshore.dist_to_ports import DistanceToPorts
from reVX.offshore.dist_to_ports_cli import main
from reVX.utilities.utilities import coordinate_distance

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


def test_dist_to_port():
    """
    Compare distance to points versus haversine distance
    """
    dtp = DistanceToPorts(PORTS_FPATH, EXCL_H5)
    test = dtp.least_cost_distance(max_workers=1).ravel()
    mask = test != -1

    cols = get_lat_lon_cols(dtp.ports)
    ports_coords = dtp.ports[cols].values

    with ExclusionLayers(EXCL_H5) as f:
        lat = f.latitude
        lon = f.longitude

    pixel_coords = np.dstack((lat.ravel(), lon.ravel()))[0]

    for p in ports_coords:
        p = np.expand_dims(p, 0)
        dist_to_ports = coordinate_distance(p, pixel_coords)
        msg = 'Least cost distance to port is less than haversine distance!'
        check = test[mask] > dist_to_ports[mask]
        assert np.all(check), msg


def test_dist_to_ports():
    """
    Compare distance to points versus haversine distance
    """
    dtp = DistanceToPorts(PORTS_FPATH, EXCL_H5)
    test = dtp.least_cost_distance(max_workers=1).ravel()

    cols = get_lat_lon_cols(dtp.ports)
    ports_coords = dtp.ports[cols].values

    with ExclusionLayers(EXCL_H5) as f:
        lat = f.latitude
        lon = f.longitude

    pixel_coords = np.dstack((lat.ravel(), lon.ravel()))[0]

    dist_to_ports = np.full((len(pixel_coords), ), np.finfo('float32').max,
                            dtype='float32')
    for p in ports_coords:
        p = np.expand_dims(p, 0)
        dist_to_ports = np.minimum(dist_to_ports,
                                   coordinate_distance(p, pixel_coords))

    mask = test != -1
    msg = 'Least cost distance to ports is less than haversine distance!'
    check = test[mask] > dist_to_ports[mask]
    print(np.min(test[mask][~check] - dist_to_ports[mask][~check]))
    print(ports_coords, pixel_coords[mask][~check])
    check = np.allclose(test[mask][~check], dist_to_ports[mask][~check])
    assert check, msg


@pytest.mark.parametrize('max_workers', [None, 1])
def test_baseline(max_workers):
    """
    Compute distance to ports
    """
    baseline = get_dist_to_ports(EXCL_H5)
    test = DistanceToPorts.run(PORTS_FPATH, EXCL_H5, max_workers=max_workers)

    msg = 'distance to ports does not match baseline distances'
    assert np.allclose(baseline, test), msg


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
