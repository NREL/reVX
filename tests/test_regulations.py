# -*- coding: utf-8 -*-
# pylint: disable=all
"""
Regulations tests
"""
import numpy as np
import pandas as pd
import os
import pytest
import tempfile

from reVX import TESTDATADIR
from reVX.utilities.regulations import AbstractBaseRegulations


GENERIC_REG_VAL = 10
REGS_FPATH = os.path.join(TESTDATADIR, 'setbacks', 'ri_wind_regs_fips.csv')


class TestRegulations(AbstractBaseRegulations):
    """Implementation of AbstractBaseRegulations for testing only."""

    def _county_regulation_value(self, __):
        """Retrieve county regulation setback. """
        return 0


def test_regulations_init():
    """Test initializing a normal regulations file. """
    regs = TestRegulations(generic_regulation_value=GENERIC_REG_VAL,
                           regulations_fpath=REGS_FPATH)
    assert np.isclose(regs.generic, GENERIC_REG_VAL)

    for col in regs.required_columns:
        assert col in regs.df
        assert not regs.df[col].isna().any()

    assert regs.df['Feature Type'].str.islower().all()
    assert regs.df['Value Type'].str.islower().all()


def test_regulations_missing_init():
    """Test initializing base regulations with missing info. """
    with pytest.raises(RuntimeError) as excinfo:
        TestRegulations()

    expected_err_msg = ('Regulations require a local regulation.csv file '
                        'and/or a generic regulation value!')
    assert expected_err_msg in str(excinfo.value)


def test_regulations_non_capitalized_cols():
    """Test base regulations for csv with non-capitalized cols. """
    regs_path = os.path.join(TESTDATADIR, 'setbacks', 'non_standard_regs',
                             'col_names_not_caps.csv')

    regs = TestRegulations(generic_regulation_value=GENERIC_REG_VAL,
                           regulations_fpath=regs_path)
    for col in regs.df.columns:
        if col.lower() not in {"geometry", "fips"}:
            assert col.istitle()


def test_regulations_missing_cols():
    """Test base regulations for csv with missing cols. """
    expected_err_msg = 'Regulations are missing the following required columns'

    for fn in ['missing_ft.csv', 'missing_vt.csv', 'missing_vt.csv']:
        regs_path = os.path.join(TESTDATADIR, 'setbacks', 'non_standard_regs',
                                 fn)

        with pytest.raises(RuntimeError) as excinfo:
            TestRegulations(generic_regulation_value=GENERIC_REG_VAL,
                            regulations_fpath=regs_path)
        assert expected_err_msg in str(excinfo.value)


def test_regulations_na_cols():
    """Test base regulations for csv with cols containing NaN's. """

    for fn in ['nan_feature_types.csv', 'nan_fips.csv', 'nan_value_types.csv',
               'nan_values.csv']:
        regs_path = os.path.join(TESTDATADIR, 'setbacks', 'non_standard_regs',
                                 fn)
        regs = TestRegulations(generic_regulation_value=GENERIC_REG_VAL,
                               regulations_fpath=regs_path)

        regs_df = pd.read_csv(regs_path)
        assert regs_df[regs.required_columns].isna().values.any()

        for col in regs.required_columns:
            assert not regs.df[col].isna().any()


def test_regulations_iter():
    """Test base regulations iterator. """
    regs_path = os.path.join(TESTDATADIR, 'setbacks',
                             'ri_parcel_regs_multiplier_solar.csv')

    regs = TestRegulations(generic_regulation_value=GENERIC_REG_VAL,
                           regulations_fpath=regs_path)
    for ind, (setback, cnty) in enumerate(regs):
        assert np.isclose(setback, 0)
        assert regs.df.iloc[[ind]].equals(cnty)

    regs = TestRegulations(generic_regulation_value=GENERIC_REG_VAL,
                           regulations_fpath=None)
    assert len(list(regs)) == 0


def test_regulations_set_to_none():
    """Test setting regulations to `None` not allowed. """
    regs = TestRegulations(generic_regulation_value=GENERIC_REG_VAL,
                           regulations_fpath=REGS_FPATH)
    with pytest.raises(ValueError):
        regs.df = None


def test_regulations_locals_exist():
    """Test locals_exist property. """
    regs = TestRegulations(generic_regulation_value=GENERIC_REG_VAL,
                           regulations_fpath=REGS_FPATH)
    assert regs.locals_exist
    regs = TestRegulations(generic_regulation_value=GENERIC_REG_VAL,
                           regulations_fpath=None)
    assert not regs.locals_exist

    with tempfile.TemporaryDirectory() as td:
        regs = pd.read_csv(REGS_FPATH).iloc[0:0]
        regulations_fpath = os.path.basename(REGS_FPATH)
        regulations_fpath = os.path.join(td, regulations_fpath)
        regs.to_csv(regulations_fpath, index=False)
        regs = TestRegulations(generic_regulation_value=GENERIC_REG_VAL,
                               regulations_fpath=regulations_fpath)
        assert not regs.locals_exist


def test_regulations_generic_exists():
    """Test generic_exists property. """
    regs = TestRegulations(generic_regulation_value=GENERIC_REG_VAL,
                           regulations_fpath=REGS_FPATH)
    assert regs.generic_exists
    regs = TestRegulations(generic_regulation_value=GENERIC_REG_VAL,
                           regulations_fpath=None)
    assert regs.generic_exists
    regs = TestRegulations(generic_regulation_value=None,
                           regulations_fpath=REGS_FPATH)
    assert not regs.generic_exists


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
