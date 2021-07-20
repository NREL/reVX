# -*- coding: utf-8 -*-
"""
Module to handle least cost xmission layers
"""
import logging

from reV.handlers.exclusions import ExclusionLayers

logger = logging.getLogger(__name__)


class XmissionCostsLayers(ExclusionLayers):
    """
    Handler for Transmission cost layers
    """
