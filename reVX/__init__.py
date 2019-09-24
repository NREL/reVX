# -*- coding: utf-8 -*-
"""
Renewable Energy Potential(V) EXchange Tool (reVX)
"""
from __future__ import print_function, division, absolute_import
import os
from reVX.resource.resource import NSRDBX, WindX
import reVX.rpm as rev_rpm
from reVX.version import __version__

__author__ = """Michael Rossol"""
__email__ = "michael.rossol@nrel.gov"

REVXDIR = os.path.dirname(os.path.realpath(__file__))
TESTDATADIR = os.path.join(os.path.dirname(REVXDIR), 'tests', 'data')
