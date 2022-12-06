# -*- coding: utf-8 -*-
"""
Compute setbacks exclusions
"""
from warnings import warn
import logging

from reVX.utilities.regulations import AbstractBaseRegulations


logger = logging.getLogger(__name__)


class FlickerRegulations(AbstractBaseRegulations):
    """Shadow flicker regulation values. """

    def __init__(self, hub_height, rotor_diameter, flicker_threshold=None,
                 regulations_fpath=None):
        """
        Parameters
        ----------
        hub_height : float | int
            Turbine hub height (m).
        rotor_diameter : float | int
            Turbine rotor diameter (m).
        flicker_threshold : float | int, optional
            Maximum number of allowable flicker hours per year to use
            for generic flicker regulations. If `None`, then only local
            (county) flicker regulations are applied.
            By default, `None`.
        regulations_fpath : str, optional
            Path to regulations .csv or .gpkg file. At a minimum, this
            file must contain the following columns: `Feature Type`
            which labels the type of regulation that each row
            represents (flicker regulations must be called "Shadow
            Flicker"), `Value Type`, which specifies the type of the
            value (flicker value types must be "Hrs/Year"), `Value`,
            which specifies the numeric value of the flicker threshold
            (in hours), and `FIPS`, which specifies a unique 5-digit
            code for each county (this can be an integer - no leading
            zeros required). If this input is `None`, generic flicker
            regulations defined by `flicker_threshold` are applied.
            By default `None`.
        """
        self._hub_height = hub_height
        self._rotor_diameter = rotor_diameter
        super().__init__(generic_regulation_value=flicker_threshold,
                         regulations_fpath=regulations_fpath)

    @property
    def hub_height(self):
        """float | int: Turbine hub-height in meters. """
        return self._hub_height

    @property
    def rotor_diameter(self):
        """float | int: Turbine rotor diameter in meters. """
        return self._rotor_diameter

    def _county_regulation_value(self, county_regulations):
        """Retrieve county regulation value. """
        regulation_type = county_regulations["Value Type"]
        regulation = float(county_regulations["Value"])
        if regulation_type != "hrs/year":
            msg = ('Cannot create flicker regulations for {}, expecting '
                   '"Hrs/Year", but got {!r}'
                   .format(county_regulations["County"], regulation_type))
            logger.warning(msg)
            warn(msg)
            return
        return regulation
