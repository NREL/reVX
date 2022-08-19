
# -*- coding: utf-8 -*-
"""
Compute setbacks
"""
from .parcel_setbacks import ParcelSetbacks
from .rail_setbacks import RailSetbacks
from .road_setbacks import RoadSetbacks
from .structure_setbacks import StructureSetbacks
from .transmission_setbacks import TransmissionSetbacks
from .water_setbacks import WaterSetbacks

SETBACKS = {'structure': StructureSetbacks,
            'road': RoadSetbacks,
            'rail': RailSetbacks,
            'transmission': TransmissionSetbacks,
            'parcel': ParcelSetbacks,
            'water': WaterSetbacks}
