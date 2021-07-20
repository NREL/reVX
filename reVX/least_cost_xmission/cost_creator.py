# -*- coding: utf-8 -*-
"""
Module to build and save least cost raster layers
"""
import logging

from reVX.utilities.exclusions_converter import ExclusionsConverter

logger = logging.getLogger(__name__)


class XmissionCostCreator(ExclusionsConverter):
    """
    Class to create and save Transmission cost layers to a .h5 Exclusion file
    """
