""" Testing these guys out

Created on Fri Jan 29 09:34:57 2021

@author: twillia2
"""
from wind_setbacks import BaseWindSetbacks, StructureWindSetbacks


excl_h5 = "/projects/rev/data/exclusions/backup/ATB_Exclusions.h5"
structures_dir = features_dir = "/projects/rev/data/conus/microsoft"
layer = layer_name = "setbacks_structures_test"
hub_height = 135
rotor_diameter = 200
regs_fpath = "/projects/rev/data_prep/setbacks/csv/county_wind_regs.csv"
# regs_fpath = None
multiplier = 1.1
chunks=(128, 128)
max_workers=1
description="Setbacks Test"
replace=False
grouby = "State"

regs_fpath = None
self = StructureWindSetbacks(excl_h5, hub_height, rotor_diameter,
                             regs_fpath=regs_fpath, multiplier=multiplier,
                             hsds=False, chunks=chunks)


features_state_map =  self._map_features_dir(features_dir)
features = features_state_map.values()

# compute_setbacks

# _generic setbacks

# _compute_generic_setbacks
features_fpath = list(features)[0]
crs = self._profile['crs']
setback = self.generic_setback
cls = StructureWindSetbacks
setbacks = cls._compute_generic_setbacks(features_fpath, crs, setback)

#_rasterize_setbacks
array = self._rasterize_setbacks(setbacks)[0]




regs_fpath = "/projects/rev/data_prep/setbacks/csv/county_wind_regs.csv"
self = StructureWindSetbacks(excl_h5, hub_height, rotor_diameter,
                             regs_fpath=regs_fpath, multiplier=multiplier,
                             hsds=False, chunks=chunks)


features_state_map =  self._map_features_dir(features_dir)
features = features_state_map.values()

# compute_setbacks

# _local setbacks

# _compute_local_setbacks
features_fpath = '/projects/rev/data/conus/microsoft/Arizona.geojson'
crs = self._profile['crs']
setback = self.generic_setback
cls = StructureWindSetbacks
wind_regs = self._regs[self._regs["State"] == "Arizona"]
setbacks = cls._compute_local_setbacks(features_fpath, crs, wind_regs,
                                       hub_height, rotor_diameter, setback)
