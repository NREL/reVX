# -*- coding: utf-8 -*-
"""
WIND Toolkit (WTK) Extraction tool
"""
import logging
from reV.handlers.resource import WindResource

from reVX.resource.base import ResourceX

logger = logging.getLogger(__name__)


class WTKX(WindResource, ResourceX):
    """
    WIND Toolkit extraction class
    """
    def __init__(self, wtk_h5, tree=None, **kwargs):
        """
        Parameters
        ----------
        wtk_h5 : str
            Path to WIND Toolkit .h5 file of interest
        tree : str
            path to .pgz file containing pickled cKDTree of lat, lon
            coordinates
        kwargs : dict
            Kwargs for Resource
        """
        super().__init__(wtk_h5, **kwargs)
        self._tree = self._init_tree(tree)
