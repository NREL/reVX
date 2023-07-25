# -*- coding: utf-8 -*-
"""
Offshore Inputs tests
"""
from click.testing import CliRunner
import json
import numpy as np
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import tempfile
import traceback

from rex.utilities.loggers import LOGGERS
from reVX import TESTDATADIR
from reVX.offshore.offshore_inputs import OffshoreInputs
from reVX.offshore.offshore_inputs_cli import main

INPUTS_FPATH = os.path.join(TESTDATADIR, 'offshore', 'offshore.h5')
OFFSHORE_SITES = os.path.join(TESTDATADIR, 'wtk', 'ri_100_wtk_2012.h5')
BASELINE = os.path.join(TESTDATADIR, 'offshore', 'inputs_baseline.csv')
INPUT_LAYERS = {'array_efficiency': 'aeff',
                'dist_to_coast': 'dist_s_to_l',
                'ports_operations': 'dist_op_to_s',
                'ports_construction_nolimits': 'dist_p_to_s_nolimit',
                'assembly_areas': 'dist_a_to_s'}


def test_site_mapping():
    """
    Test mapping of site gids to tech map
    """
    with OffshoreInputs(INPUTS_FPATH, OFFSHORE_SITES) as inp:
        meta = inp.meta
        techmap = inp['techmap_wtk']

    msg = 'offshore site gids do not match techmap gids!'
    test = meta['gid'].values
    truth = techmap[meta['row_idx'].values, meta['col_idx'].values]
    assert np.allclose(truth, test), msg


def test_extract_inputs():
    """
    test offshore inputs extraction
    """
    baseline = pd.read_csv(BASELINE)
    test = OffshoreInputs.extract(INPUTS_FPATH, OFFSHORE_SITES,
                                  input_layers=INPUT_LAYERS)

    # pandas v2 has new nan vs. None behavior, simple fill makes them match
    baseline = baseline.fillna(value="None")
    test = test.fillna(value="None")

    assert_frame_equal(baseline, test, check_dtype=False)


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


def test_cli(runner):
    """
    Test CLI
    """

    with tempfile.TemporaryDirectory() as td:
        input_layers = os.path.join(td, 'input_layers.json')
        with open(input_layers, 'w') as f:
            json.dump({'input_layers': INPUT_LAYERS}, f)

        config = {
            "log_directory": td,
            "execution_control": {
                "option": "local"
            },
            "inputs_fpath": INPUTS_FPATH,
            "offshore_sites": OFFSHORE_SITES,
            "input_layers": input_layers,
        }
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config',
                                      '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        test = os.path.basename(input_layers).replace('.json', '.csv')
        test = os.path.join(td, test)
        test = pd.read_csv(test)

        baseline = pd.read_csv(BASELINE)
        assert_frame_equal(baseline, test, check_dtype=False)

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
