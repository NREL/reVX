# -*- coding: utf-8 -*-
"""
WIND Toolkit (WTK) Extraction tools
"""
from reV.handlers import Resource


class WTK(Resource):
    """
    WIND Toolkit extraction class
    """
    def __init__(self, wtk_h5, tree=None, unscale=True):
        """
        Parameters
        ----------
        """
