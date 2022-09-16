# -*- coding: utf-8 -*-
"""
Compute setbacks exclusions
"""
from warnings import warn
import logging
import geopandas as gpd

from rex.utilities import parse_table


logger = logging.getLogger(__name__)


class Regulations:
    """Regulation setback values. """

    REQUIRED_COLUMNS = ["Feature Type", "Value Type", "Value", "FIPS"]

    def __init__(self, base_setback_dist, regulations_fpath=None,
                 multiplier=None):
        """
        Parameters
        ----------
        base_setback_dist : float | int
            Base setback distance (m). This value will be used to
            calculate the setback distance when a multiplier is provided
            either via the `regulations_fpath`csv or the `multiplier`
            input. In these cases, the setbacks will be calculated using
            `base_setback_dist * multiplier`.
        regulations_fpath : str | None, optional
            Path to regulations .csv or .gpkg file. At a minimum, this
            file must contain the following columns: `Feature Type`
            which labels the type of setback that each row represents,
            `Value Type`, which specifies wether the value is a
            multiplier or static height, `Value`, which specifies the
            numeric value of the setback or multiplier, and `FIPS`,
            which specifies a unique 5-digit code for each county (this
            can be an integer - no leading zeros required). Valid
            options for the `Value Type` are:
                - "Structure Height Multiplier"
                - "Meters"
            If this input is `None`, a generic setback of
            `base_setback_dist * multiplier` is used. By default `None`.
        multiplier : int | float | str | None, optional
            A setback multiplier to use if regulations are not supplied.
            This multiplier will be applied to the ``base_setback_dist``
            to calculate the setback. If supplied along with
            ``regulations_fpath``, this input will be used to apply a
            setback to all counties not listed in the regulations file.
            By default `None`.
        """

        self._base_setback_dist = base_setback_dist
        self._regulations = None
        self._multi = multiplier
        self._preflight_check(regulations_fpath)

    def _preflight_check(self, regulations_fpath):
        """Apply preflight checks to the regulations path and multiplier.

        Run preflight checks on setback inputs:
        1) Ensure either a regulations .csv or
           a setback multiplier (or both) is provided
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
                self.regulations = parse_table(regulations_fpath)
            except ValueError:
                self.regulations = gpd.read_file(regulations_fpath)
            logger.debug('Computing setbacks using regulations provided in: {}'
                         .format(regulations_fpath))

        if self._multi:
            logger.debug('Computing setbacks using base setback distance '
                         'multiplier of {}'.format(self._multi))

        if not regulations_fpath and not self._multi:
            msg = ('Computing setbacks requires a regulations '
                   '.csv file and/or a generic multiplier!')
            logger.error(msg)
            raise RuntimeError(msg)

    @property
    def regulations(self):
        """Regulations table.

        Returns
        -------
        geopandas.GeoDataFrame | None
        """
        return self._regulations

    @regulations.setter
    def regulations(self, regulations):
        if regulations is None:
            msg = "Cannot set regulations to `None`"
            logger.error(msg)
            raise ValueError(msg)
        self._regulations = regulations
        self._validate_regulations()

    def _validate_regulations(self):
        """Perform several validations on regulations"""

        self._convert_cols_to_title()
        self._check_for_req_missing_cols()
        self._remove_nans_from_req_cols()
        self._casefold_feature_types()

    def _convert_cols_to_title(self):
        """Convert column names in regulations DataFrame to str.title(). """
        new_col_names = {col: col.lower().title()
                         for col in self._regulations.columns
                         if col.lower() not in {"geometry", "fips"}}
        self._regulations = self._regulations.rename(new_col_names, axis=1)

    def _check_for_req_missing_cols(self):
        """Check for missing (required) columns in regulations DataFrame. """
        missing = [col for col in self.REQUIRED_COLUMNS
                   if col not in self._regulations]
        if any(missing):
            msg = ('Regulations are missing the following required columns: {}'
                   .format(missing))
            logger.error(msg)
            raise RuntimeError(msg)

    def _remove_nans_from_req_cols(self):
        """Remove rows with NaN values from required columns. """
        for col in self.REQUIRED_COLUMNS:
            na_rows = self._regulations[col].isna()
            self._regulations = self._regulations[~na_rows]

    def _casefold_feature_types(self):
        """Casefold "Feature Type" values. """
        feature_types = self._regulations['Feature Type'].str.strip()
        feature_types = feature_types.str.casefold()
        self._regulations['Feature Type'] = feature_types

    @property
    def base_setback_dist(self):
        """The base setback distance, in meters.

        Returns
        -------
        float
        """
        return self._base_setback_dist

    @property
    def generic(self):
        """Default regulation value.

        This value is used for global regulations.

        Returns
        -------
        float | None
        """
        if self.multiplier is None:
            setback = None
        else:
            setback = self.base_setback_dist * self.multiplier

        return setback

    @property
    def multiplier(self):
        """Generic setback multiplier.

        Returns
        -------
        int | float
        """
        return self._multi

    @property
    def locals_exist(self):
        """Flag indicating wether local regulations exist.

        Returns
        -------
        bool
        """
        return (self.regulations is not None and not self.regulations.empty)

    @property
    def generic_exists(self):
        """Flag indicating wether generic regulations exist.

        Returns
        -------
        bool
        """
        return self.generic is not None

    def __iter__(self):
        if self._regulations is None:
            return
        for ind, county_regulations in self.regulations.iterrows():
            setback = self._county_regulation_setback(county_regulations)
            if setback is None:
                continue
            yield setback, self.regulations.iloc[[ind]].copy()

    def _county_regulation_setback(self, county_regulations):
        """Retrieve county regulation setback. """
        setback_type = county_regulations["Value Type"].strip()
        setback = float(county_regulations["Value"])
        if setback_type.lower() == "structure height multiplier":
            setback *= self.base_setback_dist
        elif setback_type.lower() != "meters":
            msg = ("Cannot create setback for {}, expecting "
                   '"Structure Height Multiplier", or '
                   '"Meters", but got {!r}'
                   .format(county_regulations["County"], setback_type))
            logger.warning(msg)
            warn(msg)
            return
        return setback


class WindRegulations(Regulations):
    """Wind regulation setback values. """

    MULTIPLIERS = {'high': 3, 'moderate': 1.1}

    def __init__(self, hub_height, rotor_diameter, regulations_fpath=None,
                 multiplier=None):
        """
        Parameters
        ----------
        hub_height : float | int
            Turbine hub height (m), used along with rotor diameter to
            compute blade tip height which is used to determine setback
            distance.
        rotor_diameter : float | int
            Turbine rotor diameter (m), used along with hub height to
            compute blade tip height which is used to determine setback
            distance.
        regulations_fpath : str | None, optional
            Path to regulations .csv or .gpkg file. At a minimum, this
            file must contain the following columns: `Feature Type`
            which labels the type of setback that each row represents,
            `Value Type`, which specifies wether the value is a
            multiplier or static height, `Value`, which specifies the
            numeric value of the setback or multiplier, and `FIPS`,
            which specifies a unique 5-digit code for each county (this
            can be an integer - no leading zeros required). Valid
            options for the `Value Type` are:
                - "Max-tip Height Multiplier"
                - "Rotor-Diameter Multiplier"
                - "Hub-height Multiplier"
                - "Meters"
            If this input is `None`, a generic setback of
            `max_tip_height * multiplier` is used. By default `None`.
        multiplier : int | float | str | None, optional
            A setback multiplier to use if regulations are not supplied.
            This multiplier will be applied to the ``base_setback_dist``
            to calculate the setback. If supplied along with
            ``regulations_fpath``, this input will be used to apply a
            setback to all counties not listed in the regulations file.
            By default `None`.
        """
        self._hub_height = hub_height
        self._rotor_diameter = rotor_diameter
        max_tip_height = hub_height + rotor_diameter / 2
        super().__init__(base_setback_dist=max_tip_height,
                         regulations_fpath=regulations_fpath,
                         multiplier=multiplier)

    def _preflight_check(self, regulations_fpath):
        """ Run preflight checks on WindRegulations inputs.

        In addition to the checks performed in `Regulations`, the
        `multiplier` is converted to a float values if a string is
        input.

        Parameters
        ----------
        regulations_fpath : str | None
            Path to wind regulations .csv or .gpkg file, if None create
            global setbacks.
        """
        super()._preflight_check(regulations_fpath)
        if isinstance(self._multi, str):
            self._multi = self.MULTIPLIERS.get(self._multi)
            logger.debug('Computing setbacks using generic Max-tip Height '
                         'Multiplier of {}'.format(self._multi))

    @property
    def hub_height(self):
        """
        Turbine hub-height in meters

        Returns
        -------
        float
        """
        return self._hub_height

    @property
    def rotor_diameter(self):
        """
        Turbine rotor diameter in meters

        Returns
        -------
        float
        """
        return self._rotor_diameter

    def _county_regulation_setback(self, county_regulations):
        """Retrieve county regulation setback. """
        setback_type = county_regulations["Value Type"].strip()
        setback = float(county_regulations["Value"])
        if setback_type.lower() == "max-tip height multiplier":
            setback *= self.base_setback_dist
        elif setback_type.lower() == "rotor-diameter multiplier":
            setback *= self.rotor_diameter
        elif setback_type.lower() == "hub-height multiplier":
            setback *= self.hub_height
        elif setback_type.lower() != "meters":
            msg = ('Cannot create setback for {}, expecting '
                   '"Max-tip Height Multiplier", '
                   '"Rotor-Diameter Multiplier", '
                   '"Hub-height Multiplier", or '
                   '"Meters", but got {!r}'
                   .format(county_regulations["County"], setback_type))
            logger.warning(msg)
            warn(msg)
            return
        return setback


def validate_regulations_input(base_setback_dist=None, hub_height=None,
                               rotor_diameter=None):
    """Validate the regulations initialization input.

    Specifically, this function raises an error unless exactly one of
    the following combinations of inputs are provided:
        - base_setback_dist
        - hub_height and rotor_diameter

    Parameters
    ----------
    base_setback_dist : float | int
        Base setback distance (m). This value will be used to calculate
        the setback distance when a multiplier is provided either via
        the `regulations_fpath` csv or the `multiplier` input. In these
        cases, the setbacks will be calculated using
        `base_setback_dist * multiplier`. By default, `None`.
    hub_height : float | int
        Turbine hub height (m), used along with rotor diameter to
        compute blade tip height which is used to determine setback
        distance.  By default, `None`.
    rotor_diameter : float | int
        Turbine rotor diameter (m), used along with hub height to
        compute blade tip height which is used to determine setback
        distance. By default, `None`.

    Raises
    ------
    RuntimeError
        If not enough info is provided (all inputs are `None`), or too
        much info is given (all inputs are not `None`).
    """
    no_base_setback = base_setback_dist is None
    invalid_turbine_specs = rotor_diameter is None or hub_height is None

    not_enough_info = no_base_setback and invalid_turbine_specs
    too_much_info = not no_base_setback and not invalid_turbine_specs
    if not_enough_info or too_much_info:
        raise RuntimeError("Must provide either `base_setback_dist` or both "
                           "`rotor_diameter` and `hub_height` (but not all "
                           "three).")


def select_regulations(base_setback_dist=None, hub_height=None,
                       rotor_diameter=None, regulations_fpath=None,
                       multiplier=None):
    """Select appropriate regulations based on input.

    Parameters
    ----------
    base_setback_dist : _type_, optional
        _description_. By default, `None`.
    hub_height : float | int
        Turbine hub height (m), used along with rotor diameter to
        compute blade tip height which is used to determine setback
        distance. By default, `None`.
    rotor_diameter : float | int
        Turbine rotor diameter (m), used along with hub height to
        compute blade tip height which is used to determine setback
        distance. By default, `None`.
    regulations_fpath : str | None, optional
        Path to regulations .csv file. At a minimum, this csv must
        contain the following columns: `Feature Type` which labels
        the type of setback that each row represents, `Value Type`,
        which specifies wether the value is a multiplier or static
        height, `Value`, which specifies the numeric value of the
        setback or multiplier, and `FIPS`, which specifies a unique
        5-digit code for each county (this can be an integer - no
        leading zeros required). Valid options for the `Value Type`
        are:
            - "Structure Height Multiplier"
            - "Meters"
        If this input is `None`, a generic setback of
        `base_setback_dist * multiplier` is used. By default `None`.
    multiplier : int | float | str | None, optional
        A setback multiplier to use if regulations are not supplied.
        This multiplier will be applied to the ``base_setback_dist``
        to calculate the setback. If supplied along with
        ``regulations_fpath``, this input will be used to apply a
        setback to all counties not listed in the regulations file.
        By default `None`.

    Returns
    -------
    Regulations
        A regulations object that can be used to calculate the requested
        setback distance.
    """

    validate_regulations_input(base_setback_dist=base_setback_dist,
                               hub_height=hub_height,
                               rotor_diameter=rotor_diameter)

    if base_setback_dist is None:
        return WindRegulations(hub_height=hub_height,
                               rotor_diameter=rotor_diameter,
                               regulations_fpath=regulations_fpath,
                               multiplier=multiplier)
    else:
        return Regulations(base_setback_dist=base_setback_dist,
                           regulations_fpath=regulations_fpath,
                           multiplier=multiplier)
