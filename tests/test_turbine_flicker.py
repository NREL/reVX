# -*- coding: utf-8 -*-
"""
Turbine Flicker tests
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

from reV.handlers.exclusions import ExclusionLayers
from reVX import TESTDATADIR
from reVX.turbine_flicker.turbine_flicker import (
    TurbineFlicker,
    load_building_layer,
    flicker_fn_out,
    _create_excl_indices,
    _get_building_indices,
    _get_flicker_excl_shifts,
    _invert_shadow_flicker_arr
)
from reVX.turbine_flicker.regulations import FlickerRegulations
from reVX.turbine_flicker.turbine_flicker_cli import main
from reVX.handlers.geotiff import Geotiff
from reVX.utilities import ExclusionsConverter

pytest.importorskip('hybrid.flicker')

EXCL_H5 = os.path.join(TESTDATADIR, 'turbine_flicker', 'blue_creek_blds.h5')
RES_H5 = os.path.join(TESTDATADIR, 'turbine_flicker', 'blue_creek_wind.h5')
HUB_HEIGHT = 135
ROTOR_DIAMETER = 108
BASELINE = 'turbine_flicker'
BLD_LAYER = 'blue_creek_buildings'


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


def test_load_building_layer():
    """Test the load building layer function. """
    building_layer = load_building_layer(EXCL_H5, BLD_LAYER)
    with ExclusionLayers(EXCL_H5) as f:
        baseline = f[BLD_LAYER]
        profile = f.profile

    assert np.allclose(building_layer, baseline)

    with tempfile.TemporaryDirectory() as td:
        tiff_fp = os.path.join(td, "temp.tiff")
        ExclusionsConverter._write_geotiff(tiff_fp, profile, baseline)
        building_layer = load_building_layer(EXCL_H5, features_path=tiff_fp)
        assert np.allclose(building_layer, baseline)


@pytest.mark.parametrize('inputs', [[], [BLD_LAYER, "A fake path"]])
def test_load_building_layer_bad_input(inputs):
    """Test the load building layer function with bad inputs. """
    with pytest.raises(RuntimeError) as excinfo:
        load_building_layer(EXCL_H5, *inputs)

    assert "Must provide either `features_path` or " in str(excinfo.value)
    assert "`building_layer` (but not both)." in str(excinfo.value)


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


@pytest.mark.parametrize('flicker_threshold', [10, 30])
def test_shadow_flicker(flicker_threshold):
    """
    Test shadow_flicker
    """
    lat, lon = 39.913373, -105.220105
    wind_dir = np.zeros(8760)
    regulations = FlickerRegulations(HUB_HEIGHT, ROTOR_DIAMETER,
                                     flicker_threshold=flicker_threshold)
    building_layer = load_building_layer(EXCL_H5, BLD_LAYER)
    tf = TurbineFlicker(EXCL_H5, RES_H5, building_layer, regulations,
                        grid_cell_size=90, max_flicker_exclusion_range=4_510)
    shadow_flicker = tf._compute_shadow_flicker(lat, lon, wind_dir)

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


def test_get_building_indices():
    """Test retrieving building indices. """
    building_layer = load_building_layer(EXCL_H5, BLD_LAYER)
    row_idx, col_idx = _get_building_indices(building_layer, 0, resolution=64,
                                             building_threshold=0)
    with ExclusionLayers(EXCL_H5) as f:
        buildings = f[BLD_LAYER, 0:64, 0:64]

    assert (buildings[row_idx, col_idx] > 0).all()

# noqa: E201,E241
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
def test_turbine_flicker(max_workers):
    """
    Test Turbine Flicker
    """
    with ExclusionLayers(EXCL_H5) as f:
        baseline = f[BASELINE]

    regulations = FlickerRegulations(HUB_HEIGHT, ROTOR_DIAMETER,
                                     flicker_threshold=30)
    building_layer = load_building_layer(EXCL_H5, BLD_LAYER)
    tf = TurbineFlicker(EXCL_H5, RES_H5, building_layer, regulations,
                        resolution=64, tm_dset='techmap_wind',
                        max_flicker_exclusion_range=4540)
    test = tf.compute_flicker_exclusions(max_workers=max_workers)
    assert np.allclose(baseline, test)


def test_local_turbine_flicker():
    """
    Test Turbine Flicker for local regulations
    """
    regs_fpath = os.path.join(TESTDATADIR, 'turbine_flicker',
                              'blue_creek_regs_value.csv')
    regulations = FlickerRegulations(HUB_HEIGHT, ROTOR_DIAMETER,
                                     regulations_fpath=regs_fpath)
    building_layer = load_building_layer(EXCL_H5, BLD_LAYER)
    with tempfile.TemporaryDirectory() as td:
        excl_h5 = os.path.join(td, os.path.basename(EXCL_H5))
        shutil.copy(EXCL_H5, excl_h5)
        with ExclusionLayers(EXCL_H5) as f:
            fips = np.zeros(f.shape, dtype=np.uint32)
            fips[:10] = 39001
            ExclusionsConverter._write_layer(excl_h5, 'cnty_fips', f.profile,
                                             fips, chunks=f.chunks)

        tf = TurbineFlicker(excl_h5, RES_H5, building_layer, regulations,
                            resolution=64, tm_dset='techmap_wind',
                            max_flicker_exclusion_range=4540)
        test = tf.compute_exclusions(None, max_workers=1)

    with ExclusionLayers(EXCL_H5) as f:
        baseline = f[BASELINE]

    assert np.allclose(baseline[:10], test[:10])
    assert not np.allclose(baseline[10:], test[10:])
    assert np.allclose(test[10:], 1)


def test_local_and_generic_turbine_flicker():
    """
    Test Turbine Flicker for local + generic regulations
    """
    regs_fpath = os.path.join(TESTDATADIR, 'turbine_flicker',
                              'blue_creek_regs_value.csv')
    regulations = FlickerRegulations(HUB_HEIGHT, ROTOR_DIAMETER,
                                     flicker_threshold=100,
                                     regulations_fpath=regs_fpath)
    regulations_generic_only = FlickerRegulations(HUB_HEIGHT, ROTOR_DIAMETER,
                                                  flicker_threshold=100,
                                                  regulations_fpath=None)
    building_layer = load_building_layer(EXCL_H5, BLD_LAYER)

    tf = TurbineFlicker(EXCL_H5, RES_H5, building_layer,
                        regulations_generic_only,
                        resolution=64, tm_dset='techmap_wind',
                        max_flicker_exclusion_range=4540)
    generic_flicker = tf.compute_exclusions(None, max_workers=1)

    with tempfile.TemporaryDirectory() as td:
        excl_h5 = os.path.join(td, os.path.basename(EXCL_H5))
        shutil.copy(EXCL_H5, excl_h5)
        with ExclusionLayers(EXCL_H5) as f:
            fips = np.zeros(f.shape, dtype=np.uint32)
            fips[:10] = 39001
            ExclusionsConverter._write_layer(excl_h5, 'cnty_fips', f.profile,
                                             fips, chunks=f.chunks)

        tf = TurbineFlicker(excl_h5, RES_H5, building_layer, regulations,
                            resolution=64, tm_dset='techmap_wind',
                            max_flicker_exclusion_range=4540)
        test = tf.compute_exclusions(None, max_workers=1)

    with ExclusionLayers(EXCL_H5) as f:
        baseline = f[BASELINE]

    assert np.allclose(baseline[:10], test[:10])
    assert not np.allclose(generic_flicker[:10], test[:10])
    assert np.allclose(generic_flicker[10:], test[10:])
    assert not np.allclose(baseline[10:], test[10:])


def test_turbine_flicker_bad_building_layer_input():
    """
    Test Turbine Flicker with bad input for max_flicker_exclusion_range
    """
    regulations = FlickerRegulations(HUB_HEIGHT, ROTOR_DIAMETER,
                                     flicker_threshold=30)
    with pytest.raises(RuntimeError) as excinfo:
        TurbineFlicker(EXCL_H5, RES_H5, np.zeros((10, 10)), regulations)

    assert "Shape of building layer" in str(excinfo.value)
    assert "does not match shape of ExclusionLayers" in str(excinfo.value)


def test_turbine_flicker_bad_max_flicker_exclusion_range_input():
    """
    Test Turbine Flicker with bad input for max_flicker_exclusion_range
    """
    regulations = FlickerRegulations(HUB_HEIGHT, ROTOR_DIAMETER,
                                     flicker_threshold=30)
    building_layer = load_building_layer(EXCL_H5, BLD_LAYER)
    with pytest.raises(TypeError) as excinfo:
        TurbineFlicker(EXCL_H5, RES_H5, building_layer, regulations,
                       max_flicker_exclusion_range='abc')

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
            "tm_dset": "techmap_wind",
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
        # out_tiff = f"{BLD_LAYER}_{HUB_HEIGHT}hh_{ROTOR_DIAMETER}rd.tiff"
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
            "tm_dset": "techmap_wind",
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
        ExclusionsConverter._write_geotiff(tiff_fp, profile, building_layer)

        excl_h5 = os.path.join(td, os.path.basename(EXCL_H5))
        shutil.copy(EXCL_H5, excl_h5)
        # out_tiff = f"{BLD_LAYER}_{HUB_HEIGHT}hh_{ROTOR_DIAMETER}rd.tiff"
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
            "features_path": tiff_fp,
            "resolution": 64,
            "tm_dset": "techmap_wind",
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


def test_cli_bad_input(runner):
    """Test Turbine Flicker CLI with bad input. """

    with tempfile.TemporaryDirectory() as td:
        tiff_fp = os.path.join(td, "temp.tiff")
        config = {
            "log_directory": td,
            "excl_fpath": EXCL_H5,
            "execution_control": {
                "option": "local",
            },
            "hub_height": HUB_HEIGHT,
            "rotor_diameter": ROTOR_DIAMETER,
            "log_level": "INFO",
            "res_fpath": RES_H5,
            "building_layer": BLD_LAYER,
            "features_path": tiff_fp,
            "resolution": 64,
            "tm_dset": "techmap_wind",
            "max_flicker_exclusion_range": 4540
        }
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config', '-c', config_path])
        assert result.exit_code == 1

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
            "tm_dset": "techmap_wind",
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
        # config["out_tiff"] = os.path.join(td, out_tiff)
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
        # config["out_tiff"] = os.path.join(td, out_tiff_20d)
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
