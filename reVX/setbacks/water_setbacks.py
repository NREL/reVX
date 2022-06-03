# -*- coding: utf-8 -*-
"""
Compute setbacks exclusions
"""
import logging
import os
import geopandas as gpd


from reVX.setbacks.base import BaseSetbacks


logger = logging.getLogger(__name__)


class WaterSetbacks(BaseSetbacks):
    """Water setbacks. """
