# -*- coding: utf-8 -*-
"""
Distance to Ports tests
"""
from click.testing import CliRunner
import json
import numpy as np
import os
import pytest
import shutil
import tempfile
import traceback

from rex.utilities.loggers import LOGGERS
from rex.utilities.utilities import get_lat_lon_cols
from reV.handlers.exclusions import ExclusionLayers
from reVX import TESTDATADIR
from reVX.handlers.geotiff import Geotiff
from reVX.offshore.dist_to_ports import DistanceToPorts
from reVX.offshore.dist_to_ports_cli import main as dtp_main
from reVX.offshore.dist_to_ports_converter import DistToPortsConverter
from reVX.cli import main as revx_main
from reVX.utilities.utilities import coordinate_distance

EXCL_H5 = os.path.join(TESTDATADIR, 'offshore', 'offshore.h5')
PORTS_FPATH = os.path.join(TESTDATADIR, 'offshore', 'ports',
                           'ports_operations.shp')


def get_dist_to_port(geotiff):
    """
    Extract "truth" dist_to_port from geotiff
    """
    with Geotiff(geotiff) as tif:
        dist_to_port = tif.values

    return dist_to_port


def get_dist_to_ports(excl_h5, ports_layer='ports_operations'):
    """
    Extract "truth" distance to ports layer from exclusion .h5 file
    """
    with ExclusionLayers(excl_h5) as f:
        dist_to_ports = f[ports_layer]

    return dist_to_ports


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


def test_haversine_versus_dist_to_port():
    """
    Compare distance to points versus haversine distance
    """
    dtp = DistanceToPorts(PORTS_FPATH, EXCL_H5)
    cols = get_lat_lon_cols(dtp.ports)

    with ExclusionLayers(EXCL_H5) as f:
        lat = f.latitude
        lon = f.longitude

    pixel_coords = np.dstack((lat.ravel(), lon.ravel()))[0]
    hav_dist = np.full(lat.shape, np.finfo('float32').max, dtype='float32')
    dist_to_ports = hav_dist.copy()
    for _, port in dtp.ports.iterrows():
        port_idx = port[['row', 'col']].values
        port_dist = port['dist_to_pixel']
        port_coords = np.expand_dims(port[cols].values, 0)
        port_coords = port_coords.astype('float32')
        h_dist = coordinate_distance(port_coords, pixel_coords)
        h_dist = h_dist.reshape(lat.shape)
        l_dist = dtp.lc_dist_to_port(dtp.cost_arr, port_idx, port_dist)
        mask = l_dist != -1

        err = (l_dist[mask] - h_dist[mask]) / h_dist[mask]
        msg = ("Haversine distance is greater than least cost distance "
               "for port {}!".format(port['name']))
        assert np.all(err > -0.05), msg

        hav_dist = np.minimum(hav_dist, h_dist)
        dist_to_ports = np.minimum(dist_to_ports, l_dist)

    mask = dist_to_ports != -1
    err = (dist_to_ports[mask] - hav_dist[mask]) / hav_dist[mask]
    msg = "Haversine distance is greater than distance to closest port!"
    assert np.all(err > -0.05), msg


@pytest.mark.parametrize('max_workers', [None, 1])
def test_dist_to_ports(max_workers):
    """
    Compute distance to ports
    """

    with tempfile.TemporaryDirectory() as td:
        excl_fpath = os.path.basename(EXCL_H5)
        excl_fpath = os.path.join(td, excl_fpath)
        shutil.copy(EXCL_H5, excl_fpath)

        DistanceToPorts.run(PORTS_FPATH, EXCL_H5, td, max_workers=max_workers)

        for f in os.listdir(td):
            if f.endswith('.tif'):
                baseline = os.path.join(TESTDATADIR, 'offshore', f)
                baseline = get_dist_to_port(baseline)
                test = get_dist_to_port(os.path.join(td, f))
                msg = ('distance to {} does not match baseline distances'
                       .format(f))
                assert np.allclose(baseline, test), msg

        convert = DistToPortsConverter(excl_fpath)
        convert.write_dist_to_ports_to_h5(td, 'test')

        test = get_dist_to_ports(excl_fpath, ports_layer='test')

    baseline = get_dist_to_ports(EXCL_H5)
    msg = 'distance to ports does not match baseline distances'
    assert np.allclose(baseline, test), msg


@pytest.mark.parametrize('ports_layer', [None, 'test'])
def test_cli(runner, ports_layer):
    """
    Test CLI
    """
    with tempfile.TemporaryDirectory() as td:
        config = {
            "log_directory": td,
            "execution_control": {
                "option": "local",
                "max_workers": 1
            },
            "excl_fpath": EXCL_H5,
            "ports_fpath": PORTS_FPATH,
        }
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(dtp_main, ['from-config',
                                          '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        for f in os.listdir(td):
            if f.endswith('.tif'):
                baseline = os.path.join(TESTDATADIR, 'offshore', f)
                baseline = get_dist_to_port(baseline)
                test = get_dist_to_port(os.path.join(td, f))
                msg = ('distance to {} does not match baseline distances'
                       .format(f))
                assert np.allclose(baseline, test), msg

        excl_fpath = os.path.basename(EXCL_H5)
        excl_fpath = os.path.join(td, excl_fpath)
        shutil.copy(EXCL_H5, excl_fpath)

        if ports_layer is None:
            ports_layer = 'ports_operations'

        layers = {'layers': {ports_layer: td}}
        layers_path = os.path.join(td, 'layers.json')
        with open(layers_path, 'w') as f:
            json.dump(layers, f)

        result = runner.invoke(revx_main, ['exclusions',
                                           '-h5', excl_fpath,
                                           'layers-to-h5',
                                           '-l', layers_path,
                                           '-dtp'])

        baseline = get_dist_to_ports(EXCL_H5)
        test = get_dist_to_ports(excl_fpath, ports_layer=ports_layer)

        msg = 'distance to ports does not match baseline distances'
        assert np.allclose(baseline, test), msg

    LOGGERS.clear()


def plot():
    """
    Plot least cost distance vs haversine distance
    """
    import matplotlib.pyplot as plt

    dtp = DistanceToPorts(PORTS_FPATH, EXCL_H5)
    cols = get_lat_lon_cols(dtp.ports)

    with ExclusionLayers(EXCL_H5) as f:
        lat = f.latitude
        lon = f.longitude
        mask = f['dist_to_coast'] > 0

    pixel_coords = np.dstack((lat.ravel(), lon.ravel()))[0]

    for _, port in dtp.ports.iterrows():
        port_idx = port[['row', 'col']].values
        port_dist = port['dist_to_pixel']
        port_coords = np.expand_dims(port[cols].values, 0).astype('float32')
        h_dist = \
            coordinate_distance(port_coords, pixel_coords).reshape(lat.shape)
        l_dist = dtp.lc_dist_to_port(dtp.cost_arr, port_idx, port_dist)

        print(port)
        plt.imshow(mask)
        plt.plot(port_idx[1], port_idx[0], 'ro')
        plt.colorbar()
        plt.show()

        vmax = l_dist[mask].max()
        plt.imshow(l_dist, vmin=0, vmax=vmax, cmap='viridis')
        plt.plot(port_idx[1], port_idx[0], 'ko')
        plt.colorbar()
        plt.show()

        diff = l_dist - h_dist
        plt.imshow(diff, vmin=np.min(diff), vmax=0.09)
        plt.plot(port_idx[1], port_idx[0], 'ro')
        plt.colorbar()
        plt.show()

        err = diff / h_dist
        vmax = err[mask].max()
        vmax = 0
        plt.imshow(err, vmin=-0.05, vmax=vmax, cmap='rainbow')
        plt.plot(port_idx[1], port_idx[0], 'ko')
        plt.colorbar()
        plt.show()


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
