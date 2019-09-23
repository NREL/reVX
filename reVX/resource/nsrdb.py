# -*- coding: utf-8 -*-
"""
NSRDB Extraction tool
"""
import logging
from reV.handlers.resource import NSRDB

from reVX.resource.base import ResourceX

logger = logging.getLogger(__name__)


class NSRDBX(NSRDB, ResourceX):
    """
    NSRDB extraction class
    """
    def __init__(self, nsrdb_h5, tree=None, **kwargs):
        """
        Parameters
        ----------
        nsrdb_h5 : str
            Path to NSRDB .h5 file of interest
        tree : str
            path to .pgz file containing pickled cKDTree of lat, lon
            coordinates
        kwargs : dict
            Kwargs for Resource
        """
        super().__init__(nsrdb_h5, **kwargs)
        self._tree = self._init_tree(tree=tree)
