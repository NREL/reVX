# -*- coding: utf-8 -*-
"""
Turbine Flicker exclusions calculator
"""
# from hybrid.flicker.flicker_mismatch_grid import FlickerMismatch

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
