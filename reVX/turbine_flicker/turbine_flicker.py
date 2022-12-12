# -*- coding: utf-8 -*-
# pylint: disable=too-many-arguments,too-many-locals
"""
Turbine Flicker exclusions calculator
"""
from concurrent.futures import as_completed
import logging
import numpy as np
import os
from warnings import warn

from reV.handlers.exclusions import ExclusionLayers
from reV.supply_curve.extent import SupplyCurveExtent
from reV.supply_curve.tech_mapping import TechMapping
from reVX.handlers.geotiff import Geotiff
from reVX.wind_dirs.mean_wind_dirs_point import MeanWindDirectionsPoint
from reVX.utilities.exclusions import AbstractBaseExclusionsMerger
from rex.resource_extraction.resource_extraction import WindX
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import log_mem

logger = logging.getLogger(__name__)


class TurbineFlicker(AbstractBaseExclusionsMerger):
    """
    Class to compute turbine shadow flicker and exclude sites that will
    cause excessive flicker on building
    """
    STEPS_PER_HOUR = 1
    DEFAULT_FEATURE_OUTFILE = 'flicker.tif'

    def __init__(self, excl_fpath, res_fpath, features, regulations,
                 building_threshold=0, resolution=640, grid_cell_size=90,
                 max_flicker_exclusion_range="10x", tm_dset='techmap_wtk',
                 hsds=False):
        """
        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 file. File must contain
            "building_layer" and "tm_dset".
        res_fpath : str
            Filepath to wind resource .h5 file containing hourly wind
            direction data
        features : str
            This input should either be the name of an exclusion layer
            in `excl_fpath` or a file path to a GeoTIFF file containing
            buildings data from which turbine flicker exclusions should
            be computed.
        regulations : `FlickerRegulations`
            A `FlickerRegulations` object used to shadow flicker
            regulation values.
        building_threshold : float, optional
            Threshold for exclusion layer values to identify pixels with
            buildings, values are % of pixel containing a building.
            Threshold is NOT inclusive. By default, `0`.
        resolution : int, optional
            SC resolution, must be input in combination with gid,
            by default 640
        grid_cell_size : float, optional
            Length (m) of a side of each grid cell in `excl_fpath`.
        max_flicker_exclusion_range : float | int | str, optional
            Max distance (m) that flicker exclusions will extend in
            any of the cardinal directions. Can also be a string like
            ``"10x"`` (default), which is interpreted as 10 times the
            turbine rotor diameter. Note that increasing this value can
            lead to drastically instead memory requirements. This value
            may be increased slightly during execution (no more then the
            size of one grid cell) in order to yield odd exclusion array
            shapes.
        tm_dset : str, optional
            Dataset / layer name for wind toolkit techmap,
            by default 'techmap_wtk'
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on
            AWS behind HSDS. By default `False`.
        """
        super().__init__(excl_fpath, regulations, features=features,
                         hsds=hsds)
        self._res_h5 = res_fpath
        self._res = resolution
        self._grid_cell_size = grid_cell_size
        self._max_flicker_exclusion_range = (
            self._parse_max_flicker_exclusion_range(
                max_flicker_exclusion_range))
        self._set_max_grid_size_for_odd_shaped_arr()
        self._flicker_preflight_check(tm_dset=tm_dset)
        self._sc_points = self._get_sc_points(tm_dset=tm_dset)
        self._fips_to_gid = {}
        self.features = self.features > building_threshold

    def _parse_max_flicker_exclusion_range(self, excl_range):
        """Convert max_flicker_exclusion_range to float if necessary. """
        if isinstance(excl_range, str) and excl_range.endswith('x'):
            rd = self._regulations.rotor_diameter
            return float(excl_range.strip('x')) * rd

        if not isinstance(excl_range, (int, float)):
            try:
                excl_range = float(excl_range)
            except Exception as e:
                msg = ('max_flicker_exclusion_range must be numeric but '
                       'received: {}, {}'.format(excl_range, type(excl_range)))
                logger.error(msg)
                raise TypeError(msg) from e

        return excl_range

    def _flicker_preflight_check(self, tm_dset='techmap_wtk'):
        """
        Check to ensure building_layer and tm_dset are in exclusion .h5 file

        Parameters
        ----------
        tm_dset : str, optional
            Dataset / layer name for wind toolkit techmap,
            by default 'techmap_wtk'
        """
        with ExclusionLayers(self._excl_fpath, hsds=self._hsds) as f:
            layers = f.layers
            exclusion_shape = f.shape

        if self.features.shape != exclusion_shape:
            msg = ("Shape of building layer {} does not match shape of "
                   "ExclusionLayers {}"
                   .format(self.features.shape, exclusion_shape))
            logger.error(msg)
            raise RuntimeError(msg)

        if tm_dset not in layers:
            logger.warning('Could not find techmap "{t}" in {e}. '
                           'Creating {t} using reV TechMapping'
                           .format(t=tm_dset, e=self._excl_fpath))
            try:
                TechMapping.run(self._excl_fpath, self._res_h5,
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
            Dataset / layer name for wind toolkit techmap, by default
            'techmap_wtk'

        Returns
        -------
        points : pandas.DataFrame
            DataFrame of valid sc point gids with their latitude and longitude
            coordinates and nearest resource gid
        """
        with SupplyCurveExtent(self._excl_fpath, resolution=self._res) as sc:
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

    def _set_max_grid_size_for_odd_shaped_arr(self):
        """Set the max_flicker_exclusion_range to multiple of 0.5 grids """
        mult = np.round(self._max_flicker_exclusion_range
                        / self._grid_cell_size) + 0.5
        self._max_flicker_exclusion_range = mult * self._grid_cell_size

    @property
    def description(self):
        """str: Description to be added to excl H5."""
        return ('Pixels with value 0 are excluded as they will cause shadow '
                'flicker on buildings. Shadow flicker is computed using a '
                '{}m hub height, {}m rotor diameter turbine.'
                .format(self._regulations.hub_height,
                        self._regulations.rotor_diameter))

    def _apply_regulations_mask(self):
        """Mask regulations to only shadow flicker. """
        flicker = self._regulations.df['Feature Type'] == 'shadow flicker'

        if not flicker.any():
            msg = "Found no local flicker regulations!"
            logger.warning(msg)
            warn(msg)

        self._regulations.df = (self._regulations.df[flicker]
                                .reset_index(drop=True))
        logger.debug('Computing flicker for regulations in {} counties'
                     .format(len(self._regulations.df)))

    def _map_fips_to_gid(self):
        """Map county FIPS values to corresponding SC gids. """

        self._fips_to_gid = {}
        reg_fips = self._regulations.df.FIPS.unique()
        with SupplyCurveExtent(self._excl_fpath, resolution=self._res) as sc:
            for gid in self._sc_points.index:
                for fips in np.unique(sc.get_excl_points('cnty_fips', gid)):
                    if fips in reg_fips:
                        self._fips_to_gid.setdefault(fips, []).append(gid)

        missing_fips = set(reg_fips) - set(self._fips_to_gid)
        if missing_fips:
            msg = ("{} counties with flicker regulations were not found on "
                   "the supply curve grid ({}): {}"
                   .format(len(missing_fips), self._excl_fpath, missing_fips))
            logger.warning(msg)
            warn(msg)

    def _points(self, fips=None):
        """Points for which to compute shadow flicker. """
        if fips is None:
            gids = self._sc_points.index
        else:
            gids = self._fips_to_gid.get(fips, [])
        return [self._sc_points.loc[gid] for gid in gids]

    @property
    def no_exclusions_array(self):
        """np.array: Array representing no exclusions. """
        return np.ones(self.features.shape, dtype=np.uint8)

    def pre_process_regulations(self):
        """Reduce regulations to correct state and features. """
        self._apply_regulations_mask()
        self._map_fips_to_gid()

    def parse_features(self):
        """Get the building layer used to compute flicker exclusions. """
        if os.path.exists(self._features):
            logger.debug("Loading building data from {}"
                         .format(self._features))
            with Geotiff(self._features) as f:
                return f.values[0]

        with ExclusionLayers(self._excl_fpath, hsds=self._hsds) as f:
            if self._features not in f.layers:
                msg = ("{} is not available in {}"
                       .format(self._features, self._excl_fpath))
                logger.error(msg)
                raise RuntimeError(msg)
            logger.debug("Loading building data from {}, layer {}"
                         .format(self._excl_fpath, self._features))
            return f[self._features]

    def _local_exclusions_arguments(self, regulation_value, cnty):
        """Compile and return arguments to `compute_local_exclusions`.

        This method should return a list or tuple of extra args to be
        passed to `compute_local_exclusions`.

        Parameters
        ----------
        regulation_value : float | int
            Regulation value for county.
        cnty : geopandas.GeoDataFrame
            Regulations for a single county.

        Returns
        -------
        TurbineFlicker
            `TurbineFlicker` object used as an extra argument to
            calculate local flicker exclusions.
        """
        return (self._regulations.hub_height,
                self._regulations.rotor_diameter,
                self._points(cnty.iloc[0]['FIPS']),
                self._res_h5,
                self._max_flicker_exclusion_range,
                self._grid_cell_size,
                self.STEPS_PER_HOUR,
                self.features,
                self._res)

    @staticmethod
    def compute_local_exclusions(regulation_value, cnty, *args):
        """Compute local flicker exclusions.

        This method computes a flicker exclusion layer using the
        information about the input county.

        Parameters
        ----------
        regulation_value : float | int
            Maximum number of allowable flicker hours in county.
        cnty : geopandas.GeoDataFrame
            Regulations for a single county.
        tf : `TurbineFlicker`
            Instance of `TurbineFlicker` objects used to compute the
            flicker exclusions.

        Returns
        -------
        flicker : ndarray
            Raster array of flicker exclusions
        """
        (hub_height, rotor_diameter, points, res_fpath,
         max_flicker_exclusion_range, grid_cell_size, steps_per_hour,
         building_layer, resolution) = args
        logger.debug('- Computing flicker for county FIPS {}'
                     .format(cnty.iloc[0]['FIPS']))
        return compute_flicker_exclusions(hub_height, rotor_diameter, points,
                                          res_fpath, regulation_value,
                                          max_flicker_exclusion_range,
                                          grid_cell_size, steps_per_hour,
                                          building_layer, resolution,
                                          max_workers=1)

    def compute_generic_exclusions(self, max_workers=None):
        """Compute generic flicker exclusions.

        This method will compute a generic flicker exclusion layer.

        Parameters
        ----------
        max_workers : int, optional
            Number of workers to use for exclusions computation, if 1
            run in serial, if > 1 run in parallel with that many
            workers, if `None` run in parallel on all available cores.
            By default `None`.

        Returns
        -------
        flicker : ndarray
            Raster array of flicker exclusions
        """
        logger.info('Computing generic flicker exclusions using a threshold '
                    'of from {} hrs/year based on {}m hub height, {}m '
                    'rotor diameter turbines'
                    .format(self._regulations.generic,
                            self._regulations.hub_height,
                            self._regulations.rotor_diameter))
        return compute_flicker_exclusions(self._regulations.hub_height,
                                          self._regulations.rotor_diameter,
                                          self._points(), self._res_h5,
                                          self._regulations.generic,
                                          self._max_flicker_exclusion_range,
                                          self._grid_cell_size,
                                          self.STEPS_PER_HOUR,
                                          self.features, self._res,
                                          max_workers=max_workers)

    @classmethod
    def input_output_filenames(cls, out_dir, features_fpath, kwargs):
        """Generate pairs of input/output file names.

        Parameters
        ----------
        out_dir : str
            Path to output file directory.
        features_fpath : str
            This input should either be the name of an exclusion layer
            in `excl_fpath` or a file path to a GeoTIFF file containing
            buildings data from which turbine flicker exclusions should
            be computed.
        kwargs : dict
            Dictionary of extra keyword-argument pairs used to
            instantiate the `exclusion_class`.

        Yields
        ------
        tuple
            An input-output filename pair.
        """
        regulations = kwargs['regulations']
        fn = flicker_fn_out(regulations.hub_height, regulations.rotor_diameter)
        yield features_fpath, os.path.join(out_dir, fn)


def _get_building_indices(building_layer, gid, resolution=640):
    """Find buildings exclusion indices

    Parameters
    ----------
    building_layer : np.ndarray
        Exclusion layer containing buildings from which turbine flicker
        exclusions will be computed.
    gid : int
        SC point gid to extract buildings for.
    resolution : int, optional
        SC resolution, must be input in combination with gid.
        By default, `640`.

    Returns
    -------
    row_idx : ndarray
        Axis 0 indices of building in sc point sub-array in full
        exclusion array.
    col_idx : ndarray
        Axis 1 indices of building in sc point sub-array in full
        exclusion array.
    shape : tuple
        Full exclusion array shape.
    """
    row_slice, col_slice = MeanWindDirectionsPoint.get_agg_slices(
        gid, building_layer.shape, resolution
    )
    sc_blds = building_layer[row_slice, col_slice]
    row_idx = np.array(range(*row_slice.indices(row_slice.stop)))
    col_idx = np.array(range(*col_slice.indices(col_slice.stop)))

    bld_row_idx, bld_col_idx = np.where(sc_blds)
    return row_idx[bld_row_idx], col_idx[bld_col_idx]


def _create_excl_indices(bld_idx, flicker_shifts, shape):
    """
    Create 2D (row, col) indices of pixels to be excluded based on
    building indices and shadow flicker shifts.

    Parameters
    ----------
    bld_idx : tuple
        (row, col) indices of building onto which shadow flicker
        exclusions are to be mapped.
    flicker_shifts : tuple
        Index shifts (row, col) from building locations to exclude based
        on shadow flicker results. Shifts are based on shadow flicker
        threshold. Shadow flicker array is inverted to represent mapping
        of shadow onto buildings
    shape : tuple
        Full exclusion array shape

    Returns
    -------
    excl_row_idx : ndarray
        Row (axis 0) indices of pixels to be excluded because they will
        cause excessive shadow flicker on building in supply curve point
        gid subset
    excl_col_idx : ndarray
        Column (axis 1) indices of pixels to be excluded because they
        will cause excessive shadow flicker on building in supply curve
        point gid subset
    """
    row_idx, col_idx = bld_idx
    row_shifts, col_shifts = flicker_shifts

    excl_row_idx = (row_idx + row_shifts[:, None]).ravel()
    excl_row_idx[excl_row_idx < 0] = 0
    excl_row_idx[excl_row_idx >= shape[0]] = shape[0] - 1

    excl_col_idx = (col_idx + col_shifts[:, None]).ravel()
    excl_col_idx[excl_col_idx < 0] = 0
    excl_col_idx[excl_col_idx >= shape[1]] = shape[1] - 1

    return excl_row_idx.astype(np.int32), excl_col_idx.astype(np.int32)


def _invert_shadow_flicker_arr(shadow_flicker):
    """
    Check to ensure the shadow_flicker array is odd in shape, i.e. both
    dimensions are odd allowing for a central pixel for the turbine to
    sit on. Flip both axes to mimic the turbine sitting on each
    building. All flicker pixels will now indicate locations where a
    turbine would need to be to cause flicker on said building

    Parameters
    ----------
    shadow_flicker : ndarray
        2D array centered on the turbine with the number of flicker
        hours per "exclusion" pixel

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


def _get_flicker_excl_shifts(shadow_flicker, flicker_threshold=30):
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
    shadow_flicker = _invert_shadow_flicker_arr(shadow_flicker)

    # normalize by number of time-steps to match shadow flicker results
    flicker_threshold /= 8760
    shape = shadow_flicker.shape
    row_shifts, col_shifts = np.where(shadow_flicker > flicker_threshold)
    check = (np.any(np.isin(row_shifts, [0, shape[0] - 1]))
             or np.any(np.isin(col_shifts, [0, shape[1] - 1])))
    if check:
        msg = ("Turbine flicker appears to extend beyond the FlickerModel "
               "domain! Consider increasing the maximum flicker exclusion "
               "range.")
        logger.warning(msg)
        warn(msg)

    row_shifts -= shape[0] // 2
    col_shifts -= shape[1] // 2

    return row_shifts, col_shifts


def _compute_shadow_flicker(rotor_diameter, lat, lon, wind_dir,
                            max_flicker_exclusion_range, grid_cell_size,
                            steps_per_hour=1):
    """
    Compute shadow flicker for given location

    Parameters
    ----------
    rotor_diameter : int | float
        Rotor diameter in meters.
    lat : float
        Latitude coordinate of turbine.
    lon : float
        Longitude coordinate of turbine.
    wind_dir : ndarray
        Time-series of wind direction for turbine.
    max_flicker_exclusion_range : float | int | str
        Max distance (m) that flicker exclusions will extend in
        any of the cardinal directions.
    grid_cell_size : int
        Length (m) of a side of each grid cell in `excl_fpath`.
    steps_per_hour : int
        Number of time steps to take per hour when computing flicker.

    Returns
    -------
    shadow_flicker : ndarray
        2D array centered on the turbine with the number of flicker
        hours per "exclusion" pixel
    """
    # Import HOPP dynamically so its not a requirement
    from hybrid.flicker.flicker_mismatch_grid import FlickerMismatch

    mult = max_flicker_exclusion_range / rotor_diameter
    FlickerMismatch.diam_mult_nwe = mult
    FlickerMismatch.diam_mult_s = mult
    FlickerMismatch.steps_per_hour = steps_per_hour
    FlickerMismatch.turbine_tower_shadow = False

    assert len(wind_dir) == 8760

    shadow_flicker = FlickerMismatch(lat, lon,
                                     blade_length=rotor_diameter / 2,
                                     angles_per_step=None,
                                     wind_dir=wind_dir,
                                     gridcell_height=grid_cell_size,
                                     gridcell_width=grid_cell_size,
                                     gridcells_per_string=1)
    shadow_flicker = shadow_flicker.create_heat_maps(range(0, 8760),
                                                     ("time", ))[0]

    return shadow_flicker


def _exclude_turbine_flicker(hub_height, rotor_diameter, point, res_fpath,
                             flicker_threshold, max_flicker_exclusion_range,
                             grid_cell_size, steps_per_hour):
    """
    Exclude all pixels that will cause flicker exceeding the
    "flicker_threshold" on buildings that exist within
    supply curve point gid subset of "building_layer". Buildings are
    defined as any pixels in the array that are "truthy". In other
    words, the locations of the buildings are found using
    `np.where(building_layer)`. Shadow flicker is computed at the supply
    curve point resolution and applied to all buildings within that
    supply curve point sub-array. Excluded pixels can extend beyond the
    supply curve point gid subset, for example if a building sits at the
    edge of the subset.

    Parameters
    ----------
    hub_height : int | float
        Hub height in meters.
    rotor_diameter : int | float
        Rotor diameter in meters.
    gid : int
        Supply curve point gid to aggregate wind directions for
    res_fpath : str
        Filepath to wind resource .h5 file containing hourly wind
        direction data
    flicker_threshold : int
        Maximum number of allowable flicker hours.
    max_flicker_exclusion_range : float | int | str
        Max distance (m) that flicker exclusions will extend in
        any of the cardinal directions.
    grid_cell_size : int
        Length (m) of a side of each grid cell in `excl_fpath`.
    steps_per_hour : int
        Number of time steps to take per hour when computing flicker.

    Returns
    -------
    excl_idx : tuple
        (row, col) shifts of pixels to be excluded because they
        will cause excessive shadow flicker from building location
    """

    with WindX(res_fpath, log_vers=False) as f:
        dset = 'winddirection_{}m'.format(hub_height)
        wind_dir = f[dset, :, int(point['res_gid'])]

        # pylint: disable=unsubscriptable-object
        if len(wind_dir) == 8784:
            wind_dir = wind_dir[:-24]

    shadow_flicker = _compute_shadow_flicker(rotor_diameter, point['latitude'],
                                             point['longitude'], wind_dir,
                                             max_flicker_exclusion_range,
                                             grid_cell_size, steps_per_hour)

    thresh = flicker_threshold
    flicker_shifts = _get_flicker_excl_shifts(shadow_flicker,
                                              flicker_threshold=thresh)

    return flicker_shifts


def compute_flicker_exclusions(hub_height, rotor_diameter, points, res_fpath,
                               flicker_threshold, max_flicker_exclusion_range,
                               grid_cell_size, steps_per_hour, building_layer,
                               resolution=640, max_workers=None):
    """Compute turbine flicker exclusions.

    Exclude all pixels that will cause flicker exceeding the
    "flicker_threshold" on any building in "building_layer".
    Buildings are defined as any pixels in the array that are "truthy".
    In other words, the locations of the buildings are found using
    `np.where(building_layer)`. Shadow flicker is computed at the
    supply curve point resolution based on a turbine with
    "hub_height" (m) and applied to all buildings within that supply
    curve point sub-array.

    Parameters
    ----------
    hub_height : int | float
        Hub height in meters.
    rotor_diameter : int | float
        Rotor diameter in meters.
    points : iterable of pd.Series
        Supply curve points to calculate shadow flicker exclusions for.
        The series for each point should include, at the minimum,
        "res_gid", "latitude" and "longitude".
    res_fpath : str
        Filepath to wind resource .h5 file containing hourly wind
        direction data.
    flicker_threshold : int
        Maximum number of allowable flicker hours.
    max_flicker_exclusion_range : float | int | str
        Max distance (m) that flicker exclusions will extend in
        any of the cardinal directions.
    grid_cell_size : int
        Length (m) of a side of each grid cell in `excl_fpath`.
    steps_per_hour : int
        Number of time steps to take per hour when computing flicker.
    building_layer : np.ndarray
        Array containing building data. Any "truthy" values in this
        array are assumed to be buildings, so be sure to apply any
        appropriate threshold/mask before passing in the array.
    resolution : int, optional
        SC resolution, by default 640.
    max_workers : int, optional
        Number of workers to use. If 1 run, in serial. If `None`,
        use all available cores. By default, `None`.

    Returns
    -------
    flicker_arr : ndarray
        2D inclusion array. Pixels to exclude (0) to prevent shadow
        flicker on buildings in "building_layer
    """

    er = max_flicker_exclusion_range
    if max_workers is None:
        max_workers = os.cpu_count()

    flicker_arr = np.ones(building_layer.shape, dtype=np.uint8)
    if max_workers > 1:
        msg = ('Computing flicker exclusions based on {}m hub '
               'height turbines with {}m rotor diameters in parallel '
               'using {} workers'
               .format(hub_height, rotor_diameter, max_workers))
        logger.info(msg)

        loggers = [__name__, 'reVX', 'rex']
        with SpawnProcessPool(max_workers=max_workers, loggers=loggers) as exe:
            futures = {}
            for point in points:
                future = exe.submit(_exclude_turbine_flicker, hub_height,
                                    rotor_diameter, point, res_fpath,
                                    flicker_threshold, er, grid_cell_size,
                                    steps_per_hour)
                futures[future] = point

            for i, future in enumerate(as_completed(futures)):
                flicker_shifts = future.result()
                point = futures.pop(future)
                row_idx, col_idx = _get_building_indices(building_layer,
                                                         point.name,
                                                         resolution=resolution)
                row_idx, col_idx = _create_excl_indices((row_idx, col_idx),
                                                        flicker_shifts,
                                                        building_layer.shape)

                flicker_arr[row_idx, col_idx] = 0
                logger.info('Completed {} out of {} gids'
                            .format((i + 1), len(futures)))
                log_mem(logger)
    else:
        msg = ('Computing local flicker exclusions based on {}m hub height, '
               '{}m rotor diameter turbines in serial.'
               .format(hub_height, rotor_diameter))
        logger.info(msg)
        for i, point in enumerate(points):
            flicker_shifts = _exclude_turbine_flicker(hub_height,
                                                      rotor_diameter, point,
                                                      res_fpath,
                                                      flicker_threshold, er,
                                                      grid_cell_size,
                                                      steps_per_hour)

            row_idx, col_idx = _get_building_indices(building_layer,
                                                     point.name,
                                                     resolution=resolution)
            row_idx, col_idx = _create_excl_indices((row_idx, col_idx),
                                                    flicker_shifts,
                                                    building_layer.shape)

            flicker_arr[row_idx, col_idx] = 0
            logger.debug('Completed {} out of {} gids'
                         .format((i + 1), len(points)))
            log_mem(logger)

    return flicker_arr


def flicker_fn_out(hub_height, rotor_diameter):
    """Generate flicker tiff outfile name.

    Parameters
    ----------
    hub_height : int
        Turbine hub-height (m).
    rotor_diameter : int
        Turbine rotor diameter (m).

    Returns
    -------
    str
        Name of flicker outfile.
    """
    return "flicker_{}hh_{}rd.tif".format(hub_height, rotor_diameter)
