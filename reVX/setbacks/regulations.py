# -*- coding: utf-8 -*-
"""
Compute setbacks exclusions
"""
from warnings import warn
import logging
import geopandas as gpd

from rex.utilities import parse_table, log_mem


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
            self._regulations = self._parse_regulations(regulations_fpath)
            logger.debug('Computing setbacks using regulations provided in: {}'
                         .format(regulations_fpath))

        if self._multi:
            logger.debug('Computing setbacks using base setback distance '
                         'multiplier of {}'.format(self._multi))

        if not regulations_fpath and not self._multi:
            msg = ('Computing setbacks requires either a regulations '
                   '.csv file or a generic multiplier!')
            logger.error(msg)
            raise RuntimeError(msg)

    def _parse_regulations(self, regulations_fpath):
        """Parse regulations file.

        Parameters
        ----------
        regulations_fpath : str
            Path to regulations .csv or .gpkg file .

        Returns
        -------
        regulations: `geopandas.GeoDataFrame`
            GeoDataFrame with county level setback regulations merged
            with county geometries, use for intersecting with setback
            features.
        """
        try:
            regulations = parse_table(regulations_fpath)
        except ValueError:
            regulations = gpd.read_file(regulations_fpath)

        new_col_names = {col: col.lower().title()
                         for col in regulations.columns
                         if col.lower() not in {"geometry", "fips"}}
        regulations = regulations.rename(new_col_names, axis=1)

        missing = [col for col in self.REQUIRED_COLUMNS
                   if col not in regulations]
        if any(missing):
            msg = ('Regulations are missing the following required columns: {}'
                   .format(missing))
            logger.error(msg)
            raise RuntimeError(msg)

        for col in self.REQUIRED_COLUMNS:
            regulations = regulations[~regulations[col].isna()]

        feature_types = regulations['Feature Type'].str.strip().str.lower()
        regulations['Feature Type'] = feature_types

        return regulations

    @property
    def base_setback_dist(self):
        """The base setback distance, in meters.

        Returns
        -------
        float
        """
        return self._base_setback_dist

    @property
    def generic_setback(self):
        """Default setback of base setback distance * multiplier.

        This value is used for global setbacks.

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
    def regulations(self):
        """Regulations table.

        Returns
        -------
        geopandas.GeoDataFrame | None
        """
        return self._regulations

    def get_regulation_setback(self, county_regulations):
        """Compute the setback distance for the county.

        Compute the setback distance (in meters) from the
        county regulations or the base setback distance.

        Parameters
        ----------
        county_regulations : pandas.Series
            Pandas Series with regulations for a single county
            or feature type. At a minimum, this Series must
            contain the following columns: `Value Type`, which
            specifies wether the value is a multiplier or static height,
            `Value`, which specifies the numeric value of the setback or
            multiplier. Valid options for the `Value Type` are:
                - "Structure Height Multiplier"
                - "Meters"

        Returns
        -------
        setback : float | None
            Setback distance in meters, or `None` if the setback
            `Value Type` was not recognized.
        """

        setback_type = county_regulations["Value Type"].strip()
        setback = float(county_regulations["Value"])
        if setback_type.lower() == "structure height multiplier":
            setback *= self.base_setback_dist
        elif setback_type.lower() != "meters":
            msg = ("Cannot create setback for {}, expecting "
                   '"Structure Height Multiplier", or '
                   '"Meters", but got {}'
                   .format(county_regulations["County"], setback_type))
            logger.warning(msg)
            warn(msg)
            setback = None

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

    def get_regulation_setback(self, county_regulations):
        """
        Compute the setback distance in meters from the county
        regulations, turbine tip height or rotor diameter.

        Parameters
        ----------
        county_regulations : pandas.Series
            Pandas Series with wind regulations for a single county or
            feature type. At a minimum, this Series must
            contain the following columns: `Value Type`, which
            specifies wether the value is a multiplier or static height,
            `Value`, which specifies the numeric value of the setback or
            multiplier. Valid options for the `Value Type` are:
                - "Max-tip Height Multiplier"
                - "Rotor-Diameter Multiplier"
                - "Hub-height Multiplier"
                - "Meters"

        Returns
        -------
        setback : float | None
            setback distance in meters, None if the setback "Value Type"
            was not recognized
        """

        setback_type = county_regulations["Value Type"].strip()
        setback = county_regulations["Value"]
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
                   '"Meters", but got {}'
                   .format(county_regulations["County"], setback_type))
            logger.warning(msg)
            warn(msg)
            setback = None

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
