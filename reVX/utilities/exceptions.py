# -*- coding: utf-8 -*-
"""
Custom Exceptions and Errors for reV
"""


class reVXError(Exception):
    """
    Generic Error for reV
    """


class ExclusionsCheckError(Exception):
    """
    Error during Exclusion layer checking
    """


class GeoTiffKeyError(Exception):
    """
    KeyError for GeoTiffHandler
    """


class RPMValueError(Exception):
    """
    ValueError for RPM Pipeline
    """


class RPMTypeError(Exception):
    """
    TypeError for RPM Pipeline
    """


class RPMRuntimeError(Exception):
    """
    RuntimeError for RPM Pipeline
    """


class ReedsValueError(Exception):
    """
    ValueError for ReEDS Pipeline
    """


class ReedsKeyError(Exception):
    """
    TypeError for ReEDS Pipeline
    """


class ReedsRuntimeError(Exception):
    """
    RuntimeError for ReEDS Pipeline
    """


class SupplyCurvePointCapacityError(Exception):
    """
    Error for supply curve point with zero capacity
    """
