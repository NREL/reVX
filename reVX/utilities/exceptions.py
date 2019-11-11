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
    ValueError for RPM Pipeline
    """


class RPMRuntimeError(Exception):
    """
    ValueError for RPM Pipeline
    """
