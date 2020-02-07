# -*- coding: utf-8 -*-
"""
Aggregate powerrose and sort directions by dominance
"""
from reV.supply_curve.aggregation import Aggregation


class PowerRoseDirections(Aggregation):
    """
    Aggregate PowerRose to Supply Curve points and sort directions in order
    of prominence using using following key:
    [[0, 1, 2]
     [7, x, 3],
     [6, 5, 4]]
    """
