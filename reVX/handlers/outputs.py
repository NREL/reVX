# -*- coding: utf-8 -*-
"""
Classes to handle reVX h5 output files.
"""

from reV.handlers.outputs import Outputs as RevOutputs
from reVX.version import __version__


class Outputs(RevOutputs):
    """
    Base class to handle reVX output data in .h5 format
    """

    def set_version_attr(self):
        """Set the version attribute to the h5 file."""
        self.h5.attrs['version'] = __version__
        self.h5.attrs['package'] = 'reVX'
