# -*- coding: utf-8 -*-
"""
Turbine Flicker exclusions calculator
"""
import numpy as np
from hybrid.flicker.flicker_mismatch_grid import FlickerMismatch
from reV.handlers.exclusions import ExclusionLayers

# tower_height = 2.5 * blade_length

# flicker_max = 30 / 8760  # 30 hours

# FlickerMismatch.diam_mult_nwe = 29
# FlickerMismatch.diam_mult_s = 29
# FlickerMismatch.steps_per_hour = 1

# # run flicker calculation of just the blades
# FlickerMismatch.turbine_tower_shadow = False
# flicker_no_tower = FlickerMismatch(lat, lon,
#                                    blade_length=blade_length,
#                                    angles_per_step=None, wind_dir=wind_dir,
#                                    gridcell_height=90, gridcell_width=90,
#                                    gridcells_per_string=1)


class TurbineFlicker:
    """
    Class to compute turbine shadow flicker and exclude sites that will
    cause excessive flicker on building
    """
    STEPS_PER_HOUR = 1
    GRIDCELL_SIZE = 90
    FLICKER_ARRAY_LEN = 128

    def __init__(self, excl_fpath, res_fpath, building_layer, hub_height=100,
                 tm_dset='techmap_wtk', resolution=128):
        """
        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 file. File must contain "building_layer"
            and "tm_dset".
        res_fpath : str
            Filepath to .h5 file containing wind direction data
        building_layer : str
            Exclusion layer containing buildings from which turbine flicker
            exclusions will be computed.
        hub_height : int, optional
            Hub-height in meters to compute turbine shadow flicker for,
            by default 100
        tm_dset : str, optional
            Dataset / layer name for wind toolkit techmap,
            by default 'techmap_wtk'
        resolution : int | None, optional
            SC resolution, must be input in combination with gid,
            by default 128
        """
        self._excl_h5 = excl_fpath
        self._res_h5 = res_fpath
        self._bld_layer = building_layer
        self._hub_height = hub_height
        self._tm_dset = tm_dset
        self._resolution = resolution

        self._preflight_check()

    def __repr__(self):
        msg = ("{} from {}m turbines and {}"
               .format(self.__class__.__name__,
                       self._hub_height,
                       self._bld_layer))

        return msg

    @classmethod
    def _shadow_flicker(cls, lat, lon, blade_length, wind_dir):
        """
        Compute shadow flicker for given location

        Parameters
        ----------
        lat : [type]
            [description]
        lon : [type]
            [description]
        blade_length : [type]
            [description]
        wind_dir : [type]
            [description]
        """
        diam = int(np.round((cls.FLICKER_ARRAY_LEN * cls.GRIDCELL_SIZE)
                            / blade_length))
        FlickerMismatch.diam_mult_nwe = diam
        FlickerMismatch.diam_mult_s = diam
        FlickerMismatch.steps_per_hour = cls.GRIDCELL_SIZE
        FlickerMismatch.turbine_tower_shadow = False

        assert len(wind_dir) == 8760

        shadow_flicker = FlickerMismatch(lat, lon,
                                         blade_length=blade_length,
                                         angles_per_step=None,
                                         wind_dir=wind_dir,
                                         gridcell_height=cls.GRIDCELL_SIZE,
                                         gridcell_width=cls.GRIDCELL_SIZE,
                                         gridcells_per_string=1)

        return shadow_flicker

    def _preflight_check(self):
        """
        Check to ensure building_layer and tm_dset are in exclusion .h5 file
        """
        with ExclusionLayers(self._excl_h5) as f:
            for dset in [self._bld_layer, self._tm_dset]:
                msg = "{} is not available in {}".format(dset, self._excl_h5)
                assert dset in f, msg
