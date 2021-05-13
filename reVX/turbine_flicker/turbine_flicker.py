# -*- coding: utf-8 -*-
"""
Turbine Flicker exclusions calculator
"""
from concurrent.futures import as_completed
import logging
import numpy as np
import os
from warnings import warn

from reV.handlers.exclusions import ExclusionLayers
from reV.supply_curve.points import SupplyCurveExtent
from reV.supply_curve.tech_mapping import TechMapping
from reVX.wind_dirs.mean_wind_dirs_point import MeanWindDirectionsPoint
from reVX.utilities.exclusions_converter import ExclusionsConverter
from rex.resource_extraction.resource_extraction import WindX
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import log_mem

logger = logging.getLogger(__name__)


class TurbineFlicker:
    """
    Class to compute turbine shadow flicker and exclude sites that will
    cause excessive flicker on building
    """
    STEPS_PER_HOUR = 1
    GRIDCELL_SIZE = 90
    FLICKER_ARRAY_LEN = 65

    def __init__(self, excl_fpath, res_fpath, building_layer,
                 resolution=640, tm_dset='techmap_wtk'):
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
        tm_dset : str, optional
            Dataset / layer name for wind toolkit techmap,
            by default 'techmap_wtk'
        resolution : int, optional
            SC resolution, must be input in combination with gid,
            by default 640
        """
        self._excl_h5 = excl_fpath
        self._res_h5 = res_fpath
        self._bld_layer = building_layer
        self._res = resolution
        self._preflight_check(tm_dset=tm_dset)
        self._sc_points = self._get_sc_points(tm_dset=tm_dset)

    def __repr__(self):
        msg = ("{} from {}"
               .format(self.__class__.__name__, self._bld_layer))

        return msg

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
        # Import HOPP dynamically so its not a requirement
        from hybrid.flicker.flicker_mismatch_grid import FlickerMismatch

        mult = (cls.FLICKER_ARRAY_LEN * cls.GRIDCELL_SIZE) / 2
        mult = mult / (blade_length * 2)
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
        shadow_flicker = shadow_flicker.create_heat_maps(range(0, 8760),
                                                         ("time", ))[0]

        return shadow_flicker

    @staticmethod
    def _invert_shadow_flicker_arr(shadow_flicker):
        """
        Check to ensure the shadow_flicker array is odd in shape, i.e. both
        dimensions are odd allowing for a central pixel for the turbine to
        sit on. Flip both axes to mimic the turbine sitting on each building.
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
            Inverted 2D shadow flicker array with odd dimensions if needed.
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

        return shadow_flicker[::-1, ::-1]

    @classmethod
    def _get_flicker_excl_shifts(cls, shadow_flicker, flicker_threshold=30):
        """
        Determine locations of shadow flicker that exceed the given threshold,
        convert to row and column shifts. These are the locations turbines
        would need to in relation to building to cause flicker exceeding the
        threshold value.

        Parameters
        ----------
        shadow_flicker : ndarray
            2D array centered on the turbine with the number of flicker hours
            per "exclusion" pixel
        flicker_threshold : int, optional
            Maximum number of allowable flicker hours, by default 30

        Returns
        -------
        row_shifts : ndarray
            Shifts along axis 0 from building location to pixels to be excluded
        col_shifts : ndarray
            Shifts along axis 1 from building location to pixels to be excluded
        """
        # ensure shadow_flicker array is regularly shaped and invert for
        # mapping to buildings
        shadow_flicker = cls._invert_shadow_flicker_arr(shadow_flicker)

        # normalize by number of time-steps to match shadow flicker results
        flicker_threshold /= 8760
        shape = shadow_flicker.shape
        row_shifts, col_shifts = np.where(shadow_flicker > flicker_threshold)
        check = (np.any(np.isin(row_shifts, [0, shape[0] - 1]))
                 or np.any(np.isin(col_shifts, [0, shape[1] - 1])))
        if check:
            msg = ("Turbine flicker exceeding {} appears to extend beyond the "
                   "FlickerModel domain! Please increase the "
                   "FLICKER_ARRAY_LEN and try again!")
            logger.error(msg)
            raise RuntimeError(msg)

        row_shifts -= shape[0] // 2
        col_shifts -= shape[1] // 2

        return row_shifts, col_shifts

    @staticmethod
    def _get_building_indices(excl_fpath, building_layer, gid,
                              resolution=640, building_threshold=0):
        """
        Find buildings in sc point sub-array and convert indices to full
        exclusion indices

        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 file. File must contain "building_layer"
            and "tm_dset".
        building_layer : str
            Exclusion layer containing buildings from which turbine flicker
            exclusions will be computed.
        gid : int
            sc point gid to extract buildings for
        resolution : int, optional
            SC resolution, must be input in combination with gid,
            by default 640
        building_threshold : float, optional
            Threshold for exclusion layer values to identify pixels with
            buildings, values are % of pixel containing a building,
            by default 0

        Returns
        -------
        row_idx : ndarray
            Axis 0 indices of building in sc point sub-array in full exclusion
            array
        col_idx : ndarray
            Axis 1 indices of building in sc point sub-array in full exclusion
            array
        shape : tuple
            Full exclusion array shape
        """
        with ExclusionLayers(excl_fpath) as f:
            shape = f.shape
            row_slice, col_slice = MeanWindDirectionsPoint.get_agg_slices(
                gid, shape, resolution)

            sc_blds = f[building_layer, row_slice, col_slice]

        row_idx = np.array(range(*row_slice.indices(row_slice.stop)))
        col_idx = np.array(range(*col_slice.indices(col_slice.stop)))
        bld_row_idx, bld_col_idx = np.where(sc_blds > building_threshold)

        return row_idx[bld_row_idx], col_idx[bld_col_idx], shape

    @staticmethod
    def _create_excl_indices(bld_idx, flicker_shifts, shape):
        """
        Create 2D (row, col) indices of pixels to be excluded based on
        building indices and shadow flicker shifts.

        Parameters
        ----------
        bld_idx : tuple
            (row, col) indices of building onto which shadow flicker exclusions
            are to be mapped.
        flicker_shifts : tuple
            Index shifts (row, col) from building locations to exclude based
            on shadow flicker results. Shifts are based on shadow flicker
            threshold. Shadow flicker array is inverted to represent mapping of
            shadow onto buildings
        shape : tuple
            Full exclusion array shape

        Returns
        -------
        excl_row_idx : ndarray
            Row (axis 0) indices of pixels to be excluded because they will
            cause excessive shadow flicker on building in supply curve point
            gid subset
        excl_col_idx : ndarray
            Column (axis 1) indices of pixels to be excluded because they will
            cause excessive shadow flicker on building in supply curve point
            gid subset
        """
        row_idx, col_idx = bld_idx
        row_shifts, col_shifts = flicker_shifts

        excl_row_idx = (row_idx + row_shifts[:, None]).ravel()
        excl_row_idx[excl_row_idx < 0] = 0
        excl_row_idx[excl_row_idx >= shape[0]] = shape[0] - 1

        excl_col_idx = (col_idx + col_shifts[:, None]).ravel()
        excl_col_idx[excl_col_idx < 0] = 0
        excl_col_idx[excl_col_idx >= shape[1]] = shape[1] - 1

        return excl_row_idx.astype(np.uint32), excl_col_idx.astype(np.uint32)

    @classmethod
    def _exclude_turbine_flicker(cls, point, excl_fpath, res_fpath,
                                 building_layer, hub_height,
                                 building_threshold=0, flicker_threshold=30,
                                 resolution=640):
        """
        Exclude all pixels that will cause flicker exceeding the
        "flicker_threshold" on buildings that exist within
        supply curve point gid subset of "building_layer". Buildings
        are defined as pixels with >= the "building_threshold".
        Shadow flicker is computed at the supply curve point
        resolution and applied to all buildings within that supply curve point
        sub-array. Excluded pixels can extend beyond the supply curve point
        gid subset, for example if a building sits at the edge of the subset.

        Parameters
        ----------
        gid : int
            Supply curve point gid to aggregate wind directions for
        excl_fpath : str
            Filepath to exclusions h5 file. File must contain "tm_dset".
        res_fpath : str
            Filepath to wind resource .h5 file containing hourly wind
            direction data
        building_layer : str
            Exclusion layer containing buildings from which turbine flicker
            exclusions will be computed.
        hub_height : int
            Hub-height in meters to compute turbine shadow flicker for
        building_threshold : float, optional
            Threshold for exclusion layer values to identify pixels with
            buildings, values are % of pixel containing a building,
            by default 0
        flicker_threshold : int, optional
            Maximum number of allowable flicker hours, by default 30
        resolution : int, optional
            SC resolution, must be input in combination with gid,
            by default 640

        Returns
        -------
        excl_idx : tuple
            (row, col) indices of pixels to be excluded because they will cause
            excessive shadow flicker on building in supply curve point gid
            subset
        """
        row_idx, col_idx, shape = cls._get_building_indices(
            excl_fpath, building_layer, point.name,
            resolution=resolution, building_threshold=building_threshold)

        with WindX(res_fpath) as f:
            dset = 'winddirection_{}m'.format(hub_height)
            wind_dir = f[dset, :, int(point['res_gid'])]

            if len(wind_dir) == 8784:
                wind_dir = wind_dir[:-24]

        blade_length = hub_height / 2.5
        shadow_flicker = cls._compute_shadow_flicker(point['latitude'],
                                                     point['longitude'],
                                                     blade_length,
                                                     wind_dir)

        flicker_shifts = cls._get_flicker_excl_shifts(
            shadow_flicker, flicker_threshold=flicker_threshold)

        excl_idx = cls._create_excl_indices((row_idx, col_idx),
                                            flicker_shifts, shape)

        return excl_idx

    def _preflight_check(self, tm_dset='techmap_wtk'):
        """
        Check to ensure building_layer and tm_dset are in exclusion .h5 file

        Parameters
        ----------
        tm_dset : str, optional
            Dataset / layer name for wind toolkit techmap,
            by default 'techmap_wtk'
        """
        with ExclusionLayers(self._excl_h5) as f:
            layers = f.layers

        if self._bld_layer not in layers:
            msg = ("{} is not available in {}"
                   .format(self._bld_layer, self._excl_h5))
            logger.error(msg)
            raise RuntimeError(msg)

        if tm_dset not in layers:
            logger.warning('Could not find techmap "{t}" in {e}. '
                           'Creating {t} using reV TechMapping'
                           .format(t=tm_dset, e=self._excl_h5))
            try:
                TechMapping.run(self._excl_h5, self._res_h5,
                                dset=tm_dset)
            except Exception as e:
                logger.exception('TechMapping process failed. Received the '
                                 'following error:\n{}'.format(e))
                raise e

    def _get_sc_points(self, tm_dset='techmap_wtk'):
        """
        Get the valid sc points to run turbine flicker for

        Parameters
        ----------
        tm_dset : str, optional
            [description], by default 'techmap_wtk'

        Returns
        -------
        points : pandas.DataFrame
            DataFrame of valid sc point gids with their latitude and longitude
            coordinates and nearest resource gid
        """
        with SupplyCurveExtent(self._excl_h5, resolution=self._res) as sc:
            points = sc.points
            points['latitude'] = sc.latitude
            points['longitude'] = sc.longitude
            gids = sc.valid_sc_points(tm_dset)
            points = points.loc[gids]

        with WindX(self._res_h5) as f:
            res_gids = f.lat_lon_gid(points[['latitude', 'longitude']].values,
                                     check_lat_lon=False)
            points['res_gid'] = res_gids

        return points

    def compute_exclusions(self, hub_height, building_threshold=0,
                           flicker_threshold=30, max_workers=None,
                           out_layer=None):
        """
        Exclude all pixels that will cause flicker exceeding the
        "flicker_threshold" on any building in "building_layer". Buildings
        are defined as pixels with >= the "building_threshold value in
        "building_layer". Shadow flicker is computed at the supply curve point
        resolution based on a turbine with "hub_height" (m) and applied to all
        buildings within that supply curve point sub-array.

        Parameters
        ----------
        hub_height : int
            Hub-height in meters to compute turbine shadow flicker for
        building_threshold : float, optional
            Threshold for exclusion layer values to identify pixels with
            buildings, values are % of pixel containing a building,
            by default 0
        flicker_threshold : int, optional
            Maximum number of allowable flicker hours, by default 30
        resolution : int, optional
            SC resolution, must be input in combination with gid,
            by default 640
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        out_layer : str, optional
            Layer to save exclusions under. Layer will be saved in
            "excl_fpath", by default None

        Returns
        -------
        flicker_arr : ndarray
            2D inclusion array. Pixels to exclude (0) to prevent shadow
            flicker on buildings in "building_layer"
        """
        with ExclusionLayers(self._excl_h5) as f:
            exclusion_shape = f.shape
            profile = f.profile

        if max_workers is None:
            max_workers = os.cpu_count()

        etf_kwargs = {"building_threshold": building_threshold,
                      "flicker_threshold": flicker_threshold,
                      "resolution": self._res}
        flicker_arr = np.ones(exclusion_shape, dtype=np.uint8)
        if max_workers > 1:
            msg = ('Computing exclusions from {} based on {}m turbines '
                   'in parallel using {} workers'
                   .format(self, hub_height, max_workers))
            logger.info(msg)

            loggers = [__name__, 'reVX', 'rex']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                futures = []
                for _, point in self._sc_points.iterrows():
                    future = exe.submit(self._exclude_turbine_flicker,
                                        point, self._excl_h5, self._res_h5,
                                        self._bld_layer, hub_height,
                                        **etf_kwargs)
                    futures.append(future)

                for i, future in enumerate(as_completed(futures)):
                    row_idx, col_idx = future.result()
                    flicker_arr[row_idx, col_idx] = 0
                    logger.debug('Completed {} out of {} gids'
                                 .format((i + 1), len(futures)))
                    log_mem(logger)
        else:
            msg = ('Computing exclusions from {} based on {}m turbines in '
                   'serial'.format(self, hub_height))
            logger.info(msg)
            for i, (_, point) in enumerate(self._sc_points.iterrows()):
                row_idx, col_idx = self._exclude_turbine_flicker(
                    point, self._excl_h5, self._res_h5, self._bld_layer,
                    hub_height, **etf_kwargs)
                flicker_arr[row_idx, col_idx] = 0
                logger.debug('Completed {} out of {} gids'
                             .format((i + 1), len(self._sc_points)))
                log_mem(logger)

        if out_layer:
            logger.info('Saving flicker exclusions to {} as {}'
                        .format(self._excl_h5, out_layer))
            description = ("Pixels with value 1 will cause greater than {} "
                           "hours of flicker on buildings in {}. Shadow "
                           "flicker is computed using a {}m turbine."
                           .format(flicker_threshold, self._bld_layer,
                                   hub_height))
            ExclusionsConverter._write_layer(self._excl_h5, out_layer,
                                             profile, flicker_arr,
                                             description=description)

        return flicker_arr

    @classmethod
    def run(cls, excl_fpath, res_fpath, building_layer, hub_height,
            tm_dset='techmap_wtk', building_threshold=0,
            flicker_threshold=30, resolution=640,
            max_workers=None, out_layer=None):
        """
        Exclude all pixels that will cause flicker exceeding the
        "flicker_threshold" on any building in "building_layer". Buildings
        are defined as pixels with >= the "building_threshold value in
        "building_layer". Shadow flicker is computed at the supply curve point
        resolution based on a turbine with "hub_height" (m) and applied to all
        buildings within that supply curve point sub-array.

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
        building_threshold : float, optional
            Threshold for exclusion layer values to identify pixels with
            buildings, values are % of pixel containing a building,
            by default 0
        flicker_threshold : int, optional
            Maximum number of allowable flicker hours, by default 30
        resolution : int, optional
            SC resolution, must be input in combination with gid,
            by default 640
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        out_layer : str, optional
            Layer to save exclusions under. Layer will be saved in
            "excl_fpath", by default None

        Returns
        -------
        flicker_arr : ndarray
            2D inclusion array. Pixels to exclude (0) to prevent shadow
            flicker on buildings in "building_layer"
        """
        flicker = cls(excl_fpath, res_fpath, building_layer,
                      resolution=resolution,
                      tm_dset=tm_dset)
        out_excl = flicker.compute_exclusions(
            hub_height,
            building_threshold=building_threshold,
            flicker_threshold=flicker_threshold,
            max_workers=max_workers,
            out_layer=out_layer)

        return out_excl
