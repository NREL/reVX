# -*- coding: utf-8 -*-
"""
Renewable Energy Potential(V) EXchange Tool (reVX)
"""
from __future__ import print_function, division, absolute_import
import os

import reVX.plexos as reV_plexos
import reVX.reeds as rev_reeds
import reVX.rpm as rev_rpm

__author__ = """Michael Rossol"""
__email__ = "michael.rossol@nrel.gov"
__version__ = "0.1.1"

REVXDIR = os.path.dirname(os.path.realpath(__file__))
TESTDATADIR = os.path.join(os.path.dirname(REVXDIR), 'tests', 'data')
