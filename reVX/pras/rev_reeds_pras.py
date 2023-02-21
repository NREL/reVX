# -*- coding: utf-8 -*-
"""
Created on Tues Feb 2023

@author: bbenton
"""

from reVX.plexos.rev_reeds_plexos import PlexosAggregation


class PrasAggregation(PlexosAggregation):
    """
    Framework to aggregate reV gen profiles to PRAS node power profiles.
    This class takes as input the plexos nodes meta data (lat/lon or shape
    files), rev supply curve table, and reeds buildout table (specifying
    which rev sc points were built and at what capacity). The class
    will build power profiles for each supply curve point and then aggregate
    the sc point profiles to the nearest neighbor pras node (if pras nodes
    are defined by lat/lon) or the shape intersect pras node (if pras nodes
    are defined by shape file).
    """
