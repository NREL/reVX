# -*- coding: utf-8 -*-
"""
Abstract generic+local regulations
"""
from abc import ABC, abstractmethod
import logging
import geopandas as gpd

from rex.utilities import parse_table


logger = logging.getLogger(__name__)


class AbstractBaseRegulations(ABC):
    """ABC for county regulation values. """

    REQUIRED_COLUMNS = ["Feature Type", "Value Type", "Value", "FIPS"]

    def __init__(self, generic_regulation_value=None, regulations_fpath=None):
        """
        Parameters
        ----------
        generic_regulation_value : float | int | None, optional
            A generic regulation value to be applied where local
            regulations and/or ordinances are not given. A `None` value
            signifies that no regulation should be applied for regions
            without a local regulation. By default `None`.
        regulations_fpath : str | None, optional
            Path to regulations .csv or .gpkg file. At a minimum, this
            file must contain the following columns: `Feature Type`
            which labels the type of regulation that each row
            represents, `Value Type`, which specifies the type of the
            value (e.g. a multiplier or static height, etc.), `Value`,
            which specifies the numeric value of the regulation, and
            `FIPS`, which specifies a unique 5-digit code for each
            county (this can be an integer - no leading zeros required).
            A `None` value signifies that no local regulations should
            be applied. By default `None`.
        """

        self._generic_regulation_value = generic_regulation_value
        self._regulations_df = None
        self._preflight_check(regulations_fpath)

    def _preflight_check(self, regulations_fpath):
        """Apply preflight checks to the regulations path and multiplier.

        Run preflight checks on setback inputs:
        1) Ensure either a regulations .csv or a generic regulation
           value (or both) is provided
        2) Ensure regulations has county FIPS, map regulations to county
           geometries from exclusions .h5 file

        Parameters
        ----------
        regulations_fpath : str | None
            Path to regulations .csv file, if `None`, create global
            setbacks.
        """
        if regulations_fpath:
            try:
                self.df = parse_table(regulations_fpath)
            except ValueError:
                self.df = gpd.read_file(regulations_fpath)
            logger.debug('Found regulations provided in: {}'
                         .format(regulations_fpath))

        if (regulations_fpath is None
            and self._generic_regulation_value is None):
            msg = ('Regulations require a local regulation.csv file '
                   'and/or a generic regulation value!')
            logger.error(msg)
            raise RuntimeError(msg)

    @property
    def generic(self):
        """float | None: Regulation value used for global regulations. """
        return self._generic_regulation_value

    @property
    def df(self):
        """geopandas.GeoDataFrame | None: Regulations table. """
        return self._regulations_df

    @df.setter
    def df(self, regulations_df):
        if regulations_df is None:
            msg = "Cannot set df to `None`"
            logger.error(msg)
            raise ValueError(msg)
        self._regulations_df = regulations_df
        self._validate_regulations()

    def _validate_regulations(self):
        """Perform several validations on regulations"""

        self._convert_cols_to_title()
        self._check_for_req_missing_cols()
        self._remove_nans_from_req_cols()
        self._casefold(cols=['Feature Type', 'Value Type'])

    def _convert_cols_to_title(self):
        """Convert column names in regulations DataFrame to str.title(). """
        new_col_names = {col: col.lower().title()
                         for col in self._regulations_df.columns
                         if col.lower() not in {"geometry", "fips"}}
        self._regulations_df = self._regulations_df.rename(new_col_names,
                                                           axis=1)

    def _check_for_req_missing_cols(self):
        """Check for missing (required) columns in regulations DataFrame. """
        missing = [col for col in self.REQUIRED_COLUMNS
                   if col not in self._regulations_df]
        if any(missing):
            msg = ('Regulations are missing the following required columns: {}'
                   .format(missing))
            logger.error(msg)
            raise RuntimeError(msg)

    def _remove_nans_from_req_cols(self):
        """Remove rows with NaN values from required columns. """
        for col in self.REQUIRED_COLUMNS:
            na_rows = self._regulations_df[col].isna()
            self._regulations_df = self._regulations_df[~na_rows]

    def _casefold(self, cols):
        """Casefold column values. """
        for col in cols:
            vals = self._regulations_df[col].str.strip().str.casefold()
            self._regulations_df[col] = vals

    @property
    def locals_exist(self):
        """bool: Flag indicating wether local regulations exist. """
        return (self.df is not None and not self.df.empty)

    @property
    def generic_exists(self):
        """bool: Flag indicating wether generic regulations exist. """
        return self.generic is not None

    def __iter__(self):
        if self._regulations_df is None:
            return
        for ind, county_regulations in self.df.iterrows():
            regulation = self._county_regulation_value(county_regulations)
            if regulation is None:
                continue
            yield regulation, self.df.iloc[[ind]].copy()

    @abstractmethod
    def _county_regulation_value(self, county_regulations):
        """Retrieve county regulation value. """
        raise NotImplementedError
