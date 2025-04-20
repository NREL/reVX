# -*- coding: utf-8 -*-
"""
Turbine Flicker tests
"""
from click.testing import CliRunner
import json
import pandas as pd
import numpy as np
import os
import pytest
import shutil
import tempfile
import traceback

from rex.utilities.loggers import LOGGERS

from reV.handlers.exclusions import ExclusionLayers
from reVX import TESTDATADIR
from reVX.turbine_flicker.turbine_flicker import (
    TurbineFlicker,
    _create_excl_indices,
    _compute_shadow_flicker,
    _get_flicker_excl_shifts,
    _invert_shadow_flicker_arr
)
from reVX.turbine_flicker.regulations import FlickerRegulations
from reVX.turbine_flicker.turbine_flicker_cli import main, flicker_fn_out
from reVX.handlers.geotiff import Geotiff
from reVX.handlers.layered_h5 import LayeredH5

pytest.importorskip('hopp.simulation.technologies.layout.flicker_mismatch')

EXCL_H5 = os.path.join(TESTDATADIR, 'turbine_flicker', 'blue_creek_blds.h5')
RES_H5 = os.path.join(TESTDATADIR, 'turbine_flicker', 'blue_creek_wind.h5')
HUB_HEIGHT = 135
ROTOR_DIAMETER = 108
BASELINE = 'turbine_flicker'
BLD_LAYER = 'blue_creek_buildings'
TM = 'techmap_wind'


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


def test_flicker_regulations():
    """Test `WindSetbackRegulations` initialization and iteration. """

    regs_path = os.path.join(TESTDATADIR, 'turbine_flicker',
                             'blue_creek_regs_value.csv')
    regs = FlickerRegulations(hub_height=100, rotor_diameter=50,
                              flicker_threshold=30,
                              regulations_fpath=regs_path)
    assert regs.hub_height == 100
    assert regs.rotor_diameter == 50

    for flicker_threshold, __ in regs:
        assert np.isclose(flicker_threshold, 30)


@pytest.mark.parametrize('shadow_loc',
                         [(2, 2),
                          (-2, -2),
                          (2, -2),
                          (-2, 2)])
def test_shadow_mapping(shadow_loc):
    """
    Test basic logic of shadow to exclusion mapping
    """
    shape = (7, 7)
    bld_idx = (np.array([3]), np.array([3]))
    baseline_row_idx = bld_idx[0] - shadow_loc[0]
    baseline_col_idx = bld_idx[1] - shadow_loc[1]
    shadow_arr = np.zeros(shape, dtype=np.int8)
    shadow_arr[bld_idx[0] + shadow_loc[0], bld_idx[1] + shadow_loc[1]] = 1

    flicker_shifts = _get_flicker_excl_shifts(shadow_arr)
    test_row_idx, test_col_idx = _create_excl_indices(bld_idx,
                                                      flicker_shifts,
                                                      shape)

    assert np.allclose(baseline_row_idx, test_row_idx)
    assert np.allclose(baseline_col_idx, test_col_idx)


def test_flicker_tech_mapping():
    """Tets that flicker runs tech mapping if it DNE. """
    regulations = FlickerRegulations(HUB_HEIGHT, ROTOR_DIAMETER,
                                     flicker_threshold=30)
    with tempfile.TemporaryDirectory() as td:
        excl_h5 = os.path.join(td, os.path.basename(EXCL_H5))
        shutil.copy(EXCL_H5, excl_h5)
        with ExclusionLayers(excl_h5) as f:
            assert "techmap_wtk" not in f.layers

        TurbineFlicker(excl_h5, RES_H5, BLD_LAYER, regulations)

        with ExclusionLayers(excl_h5) as f:
            assert "techmap_wtk" in f.layers


@pytest.mark.parametrize('flicker_threshold', [10])
def test_shadow_flicker(flicker_threshold):
    """
    Test shadow_flicker
    """
    lat, lon = 39.913373, -105.220105
    wind_dir = np.zeros(8760)
    shadow_flicker = _compute_shadow_flicker(ROTOR_DIAMETER, lat, lon,
                                             wind_dir,
                                             max_flicker_exclusion_range=4_545,
                                             grid_cell_size=90,
                                             steps_per_hour=1)

    baseline = (shadow_flicker[::-1, ::-1].copy()
                <= (flicker_threshold / 8760)).astype(np.int8)
    row_shifts, col_shifts = _get_flicker_excl_shifts(
        shadow_flicker, flicker_threshold=flicker_threshold)

    test = np.ones_like(baseline)
    test[50, 50] = 0
    test[row_shifts + 50, col_shifts + 50] = 0

    assert np.allclose(baseline, test)


def test_excl_indices_mapping():
    """
    Test mapping of shadow flicker shifts to building locations to create
    exclusion indices
    """
    shape = (129, 129)
    arr = np.random.rand(shape[0] - 2, shape[1] - 2)
    arr = np.pad(arr, 1)
    baseline = (arr <= 0.8).astype(np.int8)

    bld_idx = (np.array([64]), np.array([64]))
    flicker_shifts = _get_flicker_excl_shifts(arr[::-1, ::-1],
                                              flicker_threshold=(0.8 * 8760))

    row_idx, col_idx = _create_excl_indices(bld_idx, flicker_shifts, shape)
    test = np.ones(shape, dtype=np.int8)
    test[row_idx, col_idx] = 0

    assert np.allclose(baseline, test)


# flake8: noqa
def test_invert_shadow_flicker_arr():
    """Test inverting the shadow flicker array. """

    arr = np.array([[ 0,  1,  2,  3],
                    [ 4,  5,  6,  7],
                    [ 8,  9, 10, 11],
                    [12, 13, 14, 15]])

    expected = np.array([[10, 9, 8],
                         [ 6, 5, 4],
                         [ 2, 1, 0]])

    with pytest.warns(Warning):
        out = _invert_shadow_flicker_arr(arr)
    assert np.allclose(out, expected)


@pytest.mark.parametrize('max_workers', [None, 1])
def test_turbine_flicker_compute_exclusions(max_workers):
    """
    Test Turbine Flicker
    """
    with ExclusionLayers(EXCL_H5) as f:
        baseline = f[BASELINE]

    regulations = FlickerRegulations(HUB_HEIGHT, ROTOR_DIAMETER,
                                     flicker_threshold=30)
    tf = TurbineFlicker(EXCL_H5, RES_H5, BLD_LAYER, regulations,
                        resolution=64, tm_dset=TM,
                        max_flicker_exclusion_range=4540)
    test = tf.compute_exclusions(max_workers=max_workers)
    assert np.allclose(baseline, test)


def test_turbine_flicker_compute_exclusions_split_points():
    """
    Test Turbine Flicker with split points input
    """
    with ExclusionLayers(EXCL_H5) as f:
        baseline = f[BASELINE]

    regulations = FlickerRegulations(HUB_HEIGHT, ROTOR_DIAMETER,
                                     flicker_threshold=30)
    tf = TurbineFlicker(EXCL_H5, RES_H5, BLD_LAYER, regulations,
                        resolution=64, tm_dset=TM,
                        max_flicker_exclusion_range=4540)
    points = tf._sc_points.sample(n=tf._sc_points.shape[0],
                                  replace=False).copy()
    tf._sc_points = points.iloc[[0]].copy()
    test1 = tf.compute_exclusions()

    tf._sc_points = points.iloc[1:].copy()
    test2 = tf.compute_exclusions()

    test = np.ones_like(baseline)
    test[test1 == 0] = 0
    test[test2 == 0] = 0
    assert np.allclose(baseline, test)


def test_local_turbine_flicker():
    """
    Test Turbine Flicker for local regulations
    """
    regulations_fpath = os.path.join(TESTDATADIR, 'turbine_flicker',
                                     'blue_creek_regs_value.csv')
    regulations = FlickerRegulations(HUB_HEIGHT, ROTOR_DIAMETER,
                                     regulations_fpath=regulations_fpath)
    with tempfile.TemporaryDirectory() as td:
        excl_h5 = os.path.join(td, os.path.basename(EXCL_H5))
        shutil.copy(EXCL_H5, excl_h5)
        with ExclusionLayers(EXCL_H5) as f:
            fips = np.zeros(f.shape, dtype=np.uint32)
            fips[:10] = 39001
            lh5 = LayeredH5(excl_h5, chunks=f.chunks)
            lh5.write_layer_to_h5(fips, 'cnty_fips', f.profile)

        tf = TurbineFlicker(excl_h5, RES_H5, BLD_LAYER, regulations,
                            resolution=64, tm_dset=TM,
                            max_flicker_exclusion_range=4540)
        test = tf.compute_exclusions(max_workers=1)

    with ExclusionLayers(EXCL_H5) as f:
        baseline = f[BASELINE]

    assert np.allclose(baseline[:10], test[:10])
    assert not np.allclose(baseline[10:], test[10:])
    assert np.allclose(test[10:], 1)


def test_local_flicker_empty_regs():
    """
    Test Turbine Flicker for empty local regulations
    """
    regulations_fpath = os.path.join(TESTDATADIR, 'turbine_flicker',
                                     'blue_creek_regs_value.csv')
    with tempfile.TemporaryDirectory() as td:
        regs = pd.read_csv(regulations_fpath).iloc[0:0]
        regulations_fpath = os.path.basename(regulations_fpath)
        regulations_fpath = os.path.join(td, regulations_fpath)
        regs.to_csv(regulations_fpath, index=False)
        regulations = FlickerRegulations(HUB_HEIGHT, ROTOR_DIAMETER,
                                         regulations_fpath=regulations_fpath)

        excl_h5 = os.path.join(td, os.path.basename(EXCL_H5))
        shutil.copy(EXCL_H5, excl_h5)
        with ExclusionLayers(EXCL_H5) as f:
            fips = np.zeros(f.shape, dtype=np.uint32)
            lh5 = LayeredH5(excl_h5, chunks=f.chunks)
            lh5.write_layer_to_h5(fips, 'cnty_fips', f.profile)

        tf = TurbineFlicker(excl_h5, RES_H5, BLD_LAYER, regulations,
                            resolution=64, tm_dset=TM,
                            max_flicker_exclusion_range=4540)
        with pytest.warns(UserWarning):
            tf.compute_exclusions(max_workers=1)


def test_local_and_generic_turbine_flicker():
    """
    Test Turbine Flicker for local + generic regulations
    """
    regulations_fpath = os.path.join(TESTDATADIR, 'turbine_flicker',
                                     'blue_creek_regs_value.csv')
    regulations = FlickerRegulations(HUB_HEIGHT, ROTOR_DIAMETER,
                                     flicker_threshold=100,
                                     regulations_fpath=regulations_fpath)
    regulations_generic_only = FlickerRegulations(HUB_HEIGHT, ROTOR_DIAMETER,
                                                  flicker_threshold=100,
                                                  regulations_fpath=None)

    tf = TurbineFlicker(EXCL_H5, RES_H5, BLD_LAYER,
                        regulations_generic_only,
                        resolution=64, tm_dset=TM,
                        max_flicker_exclusion_range=4540)
    generic_flicker = tf.compute_exclusions(max_workers=1)

    with tempfile.TemporaryDirectory() as td:
        excl_h5 = os.path.join(td, os.path.basename(EXCL_H5))
        shutil.copy(EXCL_H5, excl_h5)
        with ExclusionLayers(EXCL_H5) as f:
            fips = np.zeros(f.shape, dtype=np.uint32)
            fips[:10] = 39001
            lh5 = LayeredH5(excl_h5, chunks=f.chunks)
            lh5.write_layer_to_h5(fips, 'cnty_fips', f.profile)

        tf = TurbineFlicker(excl_h5, RES_H5, BLD_LAYER, regulations,
                            resolution=64, tm_dset=TM,
                            max_flicker_exclusion_range=4540)
        test = tf.compute_exclusions(max_workers=1)

    with ExclusionLayers(EXCL_H5) as f:
        baseline = f[BASELINE]

    assert np.allclose(baseline[:10], test[:10])
    assert not np.allclose(generic_flicker[:10], test[:10])
    assert np.allclose(generic_flicker[10:], test[10:])
    assert not np.allclose(baseline[10:], test[10:])


def test_turbine_flicker_bad_max_flicker_exclusion_range_input():
    """
    Test Turbine Flicker with bad input for max_flicker_exclusion_range
    """
    regulations = FlickerRegulations(HUB_HEIGHT, ROTOR_DIAMETER,
                                     flicker_threshold=30)
    with pytest.raises(TypeError) as excinfo:
        TurbineFlicker(EXCL_H5, RES_H5, BLD_LAYER, regulations,
                       tm_dset=TM, max_flicker_exclusion_range='abc')

    assert "max_flicker_exclusion_range must be numeric" in str(excinfo.value)


def test_cli(runner):
    """
    Test Flicker CLI
    """

    with tempfile.TemporaryDirectory() as td:
        excl_h5 = os.path.join(td, os.path.basename(EXCL_H5))
        shutil.copy(EXCL_H5, excl_h5)
        out_layer = f"{BLD_LAYER}_{HUB_HEIGHT}hh_{ROTOR_DIAMETER}rd"
        config = {
            "log_directory": td,
            "excl_fpath": excl_h5,
            "execution_control": {
                "option": "local",
            },
            "building_layer": BLD_LAYER,
            "hub_height": HUB_HEIGHT,
            "out_layer": out_layer,
            "rotor_diameter": ROTOR_DIAMETER,
            "log_level": "INFO",
            "res_fpath": RES_H5,
            "resolution": 64,
            "tm_dset": TM,
            "max_flicker_exclusion_range": 4540
        }
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config', '-c', config_path])
        msg = 'Failed with error {}'.format(
            traceback.print_exception(*result.exc_info)
        )
        assert result.exit_code == 0, msg

        with ExclusionLayers(EXCL_H5) as f:
            baseline = f[BASELINE]

        with ExclusionLayers(excl_h5) as f:
            test = f[out_layer]

        assert np.allclose(baseline, test)

    LOGGERS.clear()


def test_cli_tiff(runner):
    """Test Turbine Flicker CLI for saving to tiff. """

    with tempfile.TemporaryDirectory() as td:
        excl_h5 = os.path.join(td, os.path.basename(EXCL_H5))
        shutil.copy(EXCL_H5, excl_h5)
        out_tiff = flicker_fn_out(HUB_HEIGHT, ROTOR_DIAMETER)
        config = {
            "log_directory": td,
            "excl_fpath": excl_h5,
            "execution_control": {
                "option": "local",
            },
            "building_layer": BLD_LAYER,
            "hub_height": HUB_HEIGHT,
            "rotor_diameter": ROTOR_DIAMETER,
            "log_level": "INFO",
            "res_fpath": RES_H5,
            "resolution": 64,
            "tm_dset": TM,
            "max_flicker_exclusion_range": 4540
        }
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config', '-c', config_path])
        msg = 'Failed with error {}'.format(
            traceback.print_exception(*result.exc_info)
        )
        assert result.exit_code == 0, msg

        with ExclusionLayers(EXCL_H5) as f:
            baseline = f[BASELINE]

        with ExclusionLayers(excl_h5) as f:
            assert out_tiff not in f.layers
            assert out_tiff.split('.') not in f.layers

        with Geotiff(os.path.join(td, out_tiff)) as f:
            test = f.values[0]

        assert np.allclose(baseline, test)

    LOGGERS.clear()


def test_cli_tiff_input(runner):
    """Test Turbine Flicker CLI with input building tiff. """

    with ExclusionLayers(EXCL_H5) as f:
        building_layer = f[BLD_LAYER]
        profile = f.profile
        baseline = f[BASELINE]

    with tempfile.TemporaryDirectory() as td:
        tiff_fp = os.path.join(td, "temp.tiff")
        Geotiff.write(tiff_fp, profile, building_layer)

        excl_h5 = os.path.join(td, os.path.basename(EXCL_H5))
        shutil.copy(EXCL_H5, excl_h5)
        out_tiff = flicker_fn_out(HUB_HEIGHT, ROTOR_DIAMETER)
        config = {
            "log_directory": td,
            "excl_fpath": excl_h5,
            "execution_control": {
                "option": "local",
            },
            "hub_height": HUB_HEIGHT,
            "rotor_diameter": ROTOR_DIAMETER,
            "log_level": "INFO",
            "res_fpath": RES_H5,
            "building_layer": tiff_fp,
            "resolution": 64,
            "tm_dset": TM,
            "max_flicker_exclusion_range": 4540
        }
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config', '-c', config_path])
        msg = 'Failed with error {}'.format(
            traceback.print_exception(*result.exc_info)
        )
        assert result.exit_code == 0, msg

        with ExclusionLayers(excl_h5) as f:
            assert out_tiff not in f.layers
            assert out_tiff.split('.') not in f.layers

        with Geotiff(os.path.join(td, out_tiff)) as f:
            test = f.values[0]

        assert np.allclose(baseline, test)

    LOGGERS.clear()


def test_cli_max_flicker_exclusion_range(runner):
    """Test Turbine Flicker CLI with max_flicker_exclusion_range value. """
    def_tiff_name = flicker_fn_out(HUB_HEIGHT, ROTOR_DIAMETER)
    with tempfile.TemporaryDirectory() as td:
        excl_h5 = os.path.join(td, os.path.basename(EXCL_H5))
        shutil.copy(EXCL_H5, excl_h5)
        out_tiff_def = f"{BLD_LAYER}_{HUB_HEIGHT}hh_{ROTOR_DIAMETER}rd.tiff"
        config = {
            "log_directory": td,
            "excl_fpath": excl_h5,
            "execution_control": {
                "option": "local",
            },
            "building_layer": BLD_LAYER,
            "hub_height": HUB_HEIGHT,
            "rotor_diameter": ROTOR_DIAMETER,
            "log_level": "INFO",
            "res_fpath": RES_H5,
            "resolution": 64,
            "tm_dset": TM,
            "max_flicker_exclusion_range": 4_540
        }
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config', '-c', config_path])
        msg = 'Failed with error {}'.format(
            traceback.print_exception(*result.exc_info)
        )
        assert result.exit_code == 0, msg
        shutil.move(os.path.join(td, def_tiff_name),
                    os.path.join(td, out_tiff_def))

        out_tiff_5k = f"{BLD_LAYER}_{HUB_HEIGHT}hh_{ROTOR_DIAMETER}rd_5k.tiff"
        config["max_flicker_exclusion_range"] = 5_000
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config', '-c', config_path])
        msg = 'Failed with error {}'.format(
            traceback.print_exception(*result.exc_info)
        )
        assert result.exit_code == 0, msg
        shutil.move(os.path.join(td, def_tiff_name),
                    os.path.join(td, out_tiff_5k))

        out_tiff_20d = f"{BLD_LAYER}_{HUB_HEIGHT}hh_{ROTOR_DIAMETER}rd_5d.tiff"
        config["max_flicker_exclusion_range"] = "20x"
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config', '-c', config_path])
        msg = 'Failed with error {}'.format(
            traceback.print_exception(*result.exc_info)
        )
        assert result.exit_code == 0, msg
        shutil.move(os.path.join(td, def_tiff_name),
                    os.path.join(td, out_tiff_20d))

        with ExclusionLayers(EXCL_H5) as f:
            baseline = f[BASELINE]

        with ExclusionLayers(excl_h5) as f:
            assert out_tiff_def not in f.layers
            assert out_tiff_def.split('.') not in f.layers
            assert out_tiff_5k not in f.layers
            assert out_tiff_5k.split('.') not in f.layers

        with Geotiff(os.path.join(td, out_tiff_def)) as f:
            test = f.values[0]

        with Geotiff(os.path.join(td, out_tiff_5k)) as f:
            test2 = f.values[0]

        with Geotiff(os.path.join(td, out_tiff_20d)) as f:
            test3 = f.values[0]

        assert np.allclose(baseline, test)
        assert np.allclose(baseline, test2)
        assert np.allclose(baseline, test3)
        assert np.allclose(test, test2)
        assert np.allclose(test, test3)
        assert np.allclose(test2, test3)

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
