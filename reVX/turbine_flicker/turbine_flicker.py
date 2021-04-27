# -*- coding: utf-8 -*-
"""
Turbine Flicker exclusions calculator
"""
from hybrid.flicker.flicker_mismatch_grid import FlickerMismatch
import logging
import numpy as np
from warnings import warn

from reV.handlers.exclusions import ExclusionLayers
from reV.supply_curve.points import (SupplyCurveExtent,
                                     AggregationSupplyCurvePoint)

logger = logging.getLogger(__name__)


class TurbineFlicker(SupplyCurveExtent):
    """
    Class to compute turbine shadow flicker and exclude sites that will
    cause excessive flicker on building
    """
    STEPS_PER_HOUR = 1
    GRIDCELL_SIZE = 90
    FLICKER_ARRAY_LEN = 129

    def __init__(self, excl_fpath, res_fpath, building_layer, hub_height,
                 tm_dset='techmap_wtk', resolution=128):
        """
        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 file. File must contain "building_layer"
            and "tm_dset".
        res_fpath : str
            Filepath to wind resource .h5 file containing hourly wind
            direction data
        building_layer : str
            Exclusion layer containing buildings from which turbine flicker
            exclusions will be computed.
        hub_height : int
            Hub-height in meters to compute turbine shadow flicker for
        tm_dset : str, optional
            Dataset / layer name for wind toolkit techmap,
            by default 'techmap_wtk'
        resolution : int, optional
            SC resolution, must be input in combination with gid,
            by default 128
        """
        self._excl_h5 = excl_fpath
        self._res_h5 = res_fpath
        self._bld_layer = building_layer
        self._hub_height = hub_height
        self._tm_dset = tm_dset
        self._preflight_check()

        super().__init__(excl_fpath, resolution=resolution)

    def __repr__(self):
        msg = ("{} from {}m turbines and {}"
               .format(self.__class__.__name__,
                       self._hub_height,
                       self._bld_layer))

        return msg

    @staticmethod
    def _aggregate_wind_dirs(gid, excl_fpath, res_fpath, hub_height,
                             tm_dset='techmap_wtk', resolution=128,
                             exclusion_shape=None):
        """
        [summary]

        Parameters
        ----------
        gid : int
            Supply curve point gid to aggregate wind directions for
        excl_fpath : str
            Filepath to exclusions h5 file. File must contain "tm_dset".
        res_fpath : str
            Filepath to wind resource .h5 file containing hourly wind
            direction data
        hub_height : int
            Hub-height in meters to compute turbine shadow flicker for
        tm_dset : str, optional
            Dataset / layer name for wind toolkit techmap,
            by default 'techmap_wtk'
        resolution : int, optional
            SC resolution, must be input in combination with gid,
            by default 128
        exclusion_shape : tuple, optional
            Shape of the full exclusions extent (rows, cols). Inputing this
            will speed things up considerably. by default None

        Returns
        -------
        site_meta : pandas.Series
            Meta data for supply curve point
        wind_dir : ndarray
            Hourly time-series of aggregated mean wind direction for desired
            supply curve point gid and hub-height
        """
        wind_dir_dset = 'winddirection_{}m'.format(hub_height)
        out = AggregationSupplyCurvePoint.run(
            gid,
            excl_fpath,
            res_fpath,
            tm_dset,
            wind_dir_dset,
            agg_method='mean_wind_dir',
            resolution=resolution,
            exclusion_shape=exclusion_shape)

        meta = out['meta']
        wind_dir = out[wind_dir_dset]

        # Drop last day of leap years
        if len(wind_dir) == 8784:
            wind_dir = wind_dir[:-24]

        return meta, wind_dir

    @classmethod
    def _compute_shadow_flicker(cls, lat, lon, blade_length, wind_dir):
        """
        Compute shadow flicker for given location

        Parameters
        ----------
        lat : float
            Latitude coordinate of turbine
        lon : float
            Longitude coordinate of turbine
        blade_length : float
            Turbine blade length. Hub height = 2.5 * blade length
        wind_dir : ndarray
            Time-series of wind direction for turbine

        Returns
        -------
        shadow_flicker : ndarray
            2D array centered on the turbine with the number of flicker hours
            per "exclusion" pixel
        """
        mult = (cls.FLICKER_ARRAY_LEN * cls.GRIDCELL_SIZE) / 2
        mult = int(np.ceil(mult / (blade_length * 2)))
        FlickerMismatch.diam_mult_nwe = mult
        FlickerMismatch.diam_mult_s = mult
        FlickerMismatch.steps_per_hour = cls.STEPS_PER_HOUR
        FlickerMismatch.turbine_tower_shadow = False

        assert len(wind_dir) == 8760

        shadow_flicker = FlickerMismatch(lat, lon,
                                         blade_length=blade_length,
                                         angles_per_step=None,
                                         wind_dir=wind_dir,
                                         gridcell_height=cls.GRIDCELL_SIZE,
                                         gridcell_width=cls.GRIDCELL_SIZE,
                                         gridcells_per_string=1)
        # pylint: disable=unbalanced-tuple-unpacking
        shadow_flicker, _ = shadow_flicker.create_heat_maps(range(0, 8760),
                                                            ("time", ))

        return shadow_flicker

    @staticmethod
    def _check_shadow_flicker_arr(shadow_flicker):
        """
        Check to ensure the shadow_flicker array is odd in shape, i.e. both
        dimensions are odd allowing for a central pixel for the turbine to
        sit on. Flip 0-axis to mimic the turbine sitting on each building.
        All flicker pixels will now indicate locations where a turbine would
        need to be to cause flicker on said building

        Parameters
        ----------
        shadow_flicker : ndarray
            2D array centered on the turbine with the number of flicker hours
            per "exclusion" pixel

        Returns
        -------
        shadow_flicker : ndarray
            Updated 2D shadow flicker array with odd dimensions if needed
        """
        reduce_slice = ()
        reduce_arr = False
        for s in shadow_flicker.shape:
            if s % 2:
                reduce_slice += (slice(None), )
            else:
                reduce_slice += (slice(0, -1), )
                reduce_arr = True

        if reduce_arr:
            shape_in = shadow_flicker.shape
            shadow_flicker = shadow_flicker[reduce_slice]
            msg = ('Shadow flicker array with shape {} does not have a '
                   'central pixel! Shade has been reduced to {}!'
                   .format(shape_in, shadow_flicker.shape))
            logger.warning(msg)
            warn(msg)

        return shadow_flicker[::-1]

    @classmethod
    def _threshod_flicker(cls, shadow_flicker, flicker_threshold=30):
        """
        Determine locations of shadow flicker that exceed the given threshold,
        convert to row and column shifts. These are the locations turbines
        would need to in relation to building to cause flicker exceeding the
        threshold value.

        Parameters
        ----------
        shadow_flicker : [type]
            [description]
        flicker_threshold : int, optional
            [description], by default 30

        Returns
        -------
        [type]
            [description]
        """
        # ensure shadow_flicker array is regularly shaped
        shadow_flicker = cls._check_shadow_flicker_arr(shadow_flicker)

        # normalize by number of time-steps to match shadow flicker results
        flicker_threshold /= 8760
        row_shifts, col_shifts = np.where(shadow_flicker >= flicker_threshold)
        row_shifts -= shadow_flicker.shape[0] // 2
        col_shifts -= shadow_flicker.shape[1] // 2

        return row_shifts, col_shifts

    @classmethod
    def _exclude_turbine_flicker(cls, gid, excl_fpath, res_fpath,
                                 building_layer, hub_height,
                                 bld_threshold=0.5, flicker_threshold=30,
                                 tm_dset='techmap_wtk', resolution=128):
        """
        [summary]

        Parameters
        ----------
        gid : [type]
            [description]
        excl_fpath : [type]
            [description]
        res_fpath : [type]
            [description]
        building_layer : [type]
            [description]
        hub_height : int, optional
            [description], by default 100
        tm_dset : str, optional
            [description], by default 'techmap_wtk'
        resolution : int, optional
            [description], by default 128

        Returns
        -------
        flicker_exclusions : ndarray

        """
        gid_slice = (building_layer, )
        with ExclusionLayers(excl_fpath) as f:
            shape = f.shape
            gid_slice += AggregationSupplyCurvePoint.get_agg_slices(gid,
                                                                    shape,
                                                                    resolution)
            sc_blds = f[gid_slice]

        meta, wind_dir = cls._aggregate_wind_dirs(gid,
                                                  excl_fpath,
                                                  res_fpath,
                                                  hub_height,
                                                  tm_dset=tm_dset,
                                                  resolution=resolution,
                                                  exclusion_shape=shape)
        blade_length = hub_height / 2.5
        shadow_flicker = cls._compute_shadow_flicker(meta['latitude'],
                                                     meta['longitude'],
                                                     blade_length,
                                                     wind_dir)

        row_shifts, col_shifts = cls._threshod_flicker(
            shadow_flicker, flicker_threshold=flicker_threshold)

        # TODO: shift indices to base exclusion idx from gid idx
        row_idx, col_idx = np.where(sc_blds >= bld_threshold)

        excl_row_idx = (row_idx + row_shifts[:, None]).ravel()
        excl_col_idx = (col_idx + col_shifts[:, None]).ravel()

        return excl_row_idx, excl_col_idx

    def _preflight_check(self):
        """
        Check to ensure building_layer and tm_dset are in exclusion .h5 file
        """
        with ExclusionLayers(self._excl_h5) as f:
            for dset in [self._bld_layer, self._tm_dset]:
                msg = "{} is not available in {}".format(dset, self._excl_h5)
                assert dset in f, msg
