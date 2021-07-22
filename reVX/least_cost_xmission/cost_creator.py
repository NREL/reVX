# -*- coding: utf-8 -*-
"""
Module to build and save least cost raster layers
"""
import logging
import numpy as np
from reVX.handlers.geotiff import Geotiff
from reV.handlers.exclusions import ExclusionLayers

from reVX.utilities.exclusions_converter import ExclusionsConverter
from .config import XmissionConfig, WATER_NLCD_CODE, WATER_MULT, \
    METERS_IN_MILE, NLCD_LAND_USE_CLASSES, HILL_MULT, MTN_MULT, HILL_SLOPE, \
    MTN_SLOPE, CELL_SIZE

logger = logging.getLogger(__name__)


class XmissionCostCreator(ExclusionsConverter):
    """
    Class to create and save Transmission cost layers to a .h5 Exclusion file
    - dist_to_coast (for valid sc_points)
    - base_costs
    - multiplier_*mw
    - xmission_barrier
    """
    def __init__(self, iso_regions_f, land_use_classes, iso_mults,
                 base_line_costs, iso_lookup, default_mults,
                 excl_h5='/shared-projects/rev/exclusions/ATB_Exclusions.h5',
                 slope_lyr='srtm_slope', nlcd_lyr='usa_mrlc_nlcd2011'):
        """
        Parameters
        ----------
        TODO

        iso_regions_f : str
            ISO regions raster file
        nlcd_f : str
            NLCD (land use) raster file
        slope_f : str
            Slope raster file
        land_use_classes : dict
            NLCD land use codes corresponding to use types
        iso_mults : list of dict
            Line cost multipliers for land use and terrain by ISO
        base_line_costs : dict
            Dict with per miles line costs for various capacities by ISO.
        iso_lookup : dict
            Keys are ISO names used in various dicts with keys being int codes
            use in the iso_regions_f.
        default_mults : dict
            Default line cost multipliers for areas not covered by the ISO
            regions.
        """
        logger.info('Loading all files')
        with Geotiff(iso_regions_f) as gt:
            self._iso_regions = gt.values[0]
            self._profile = gt.profile

        with ExclusionLayers(excl_h5) as el:
            self._slope = el[slope_lyr]
            self._land_use = el[nlcd_lyr]
            # self._write_geotiff('out/slope.tif', self._profile, self._slope)
            # self._write_geotiff('out/land_use.tif', self._profile, self._land_use)

        assert self._iso_regions.shape == self._land_use.shape == \
            self._slope.shape, 'All rasters must be the same shape'
        logger.info('Loading files complete')

        self._profile['dtype'] = 'float32'
        self._profile['nodata'] = None

        self._land_use_classes = land_use_classes
        self._iso_mults = iso_mults
        self._base_line_costs = base_line_costs
        self._iso_lookup = iso_lookup
        self._default_mults = default_mults
        self._cell_size = CELL_SIZE

    @classmethod
    def build_cost_rasters(cls, iso_regions_f,
                           h5_f='costs.h5',
                           land_use_classes=NLCD_LAND_USE_CLASSES,
                           iso_mults_f=None, base_line_costs_f=None,
                           iso_lookup_f=None, power_classes_f=None,
                           default_mults={}, save_to_h5=True):
        """
        Build cost rasters using base line costs and multipliers. Save to
        h5 file or tiff

        Parameters
        ----------
        iso_regions_f : str
            File with raster of ISO regions
        h5_f : str
            H5 file to save costs to
        land_use_classes : dict
            NCLD land use codes corresponding to use classes for multipliers
        iso_mults_f : None | str
            ISO multipliers JSON file. Use default if None
        base_line_costs_f : None | str
            Base line cost per mile for ISOs file. Use default if None
        iso_lookup_f : None | str
            JSON file linking ISO region code to name. Use default if None
        power_classes_f : None | str
            JSON of line capacities to use for reV power classes. Use default
            if None
        default_mults : dict
            Multipliers for regions not specified in iso_mults_f.
        save_to_h5 : bool
            Save output to h5 if true, save to geotiff if false
        """
        xc = XmissionConfig(iso_mults_f=iso_mults_f,
                            base_line_costs_f=base_line_costs_f,
                            iso_lookup_f=iso_lookup_f,
                            power_classes_f=power_classes_f)

        xcc = cls(iso_regions_f, land_use_classes, xc.iso_mults,
                  xc.base_line_costs, xc.iso_lookup, default_mults)
        mults_arr = xcc.create_multipliers()

        if save_to_h5:
            xcc._init_h5(h5_f, iso_regions_f)
        else:
            xcc._write_geotiff('out/multipliers.tif', xcc._profile, mults_arr)

        for power_class, capacity in xc.power_classes.items():
            logger.info(f'Calculating costs for class {power_class} using a '
                        f'{capacity}MW line')
            blc_arr = xcc.create_base_line_costs(capacity)

            if not save_to_h5:
                xcc._write_geotiff(f'out/blc_{capacity}MW.tif',
                                   xcc._profile, blc_arr)

            costs_arr = blc_arr * mults_arr
            if save_to_h5:
                xcc._write_layer(h5_f, f'tie_line_costs_{capacity}MW',
                                 xcc._profile, costs_arr)
            else:
                xcc._write_geotiff(f'out/tif_line_costs{capacity}MW.tif',
                                   xcc._profile, costs_arr)

    def create_multipliers(self):
        """
        Create costs multiplier raster

        Returns
        -------
        numpy.ndarray
            Costs multiplier raster, including slope and land use
        """
        mults_arr = np.ones(self._iso_regions.shape, dtype=np.float32)
        regions_mask = np.full(mults_arr.shape, False, dtype=bool)

        for r_conf in self._iso_mults:
            iso = r_conf['iso']
            logger.info(f'Processing multipliers for region {iso}')

            if self._iso_lookup is not None:
                iso_name = iso
                iso = self._iso_lookup[iso]
                logger.debug(f'{iso_name} has id {iso}')

            mask = self._iso_regions == iso
            regions_mask = regions_mask | mask

            if 'land_use' in r_conf:
                r_lu = self._land_use[mask]
                lu_mult = self._create_land_use_mult(r_lu, r_conf['land_use'])
                mults_arr[mask] = lu_mult

            if 'slope' in r_conf:
                r_slope = self._slope[mask]
                slope_mult = self._create_slope_mult(r_slope, r_conf['slope'])
                mults_arr[mask] = mults_arr[mask] * slope_mult

        # Calculate multipliers for regions not defined in `config`
        logger.debug('Processing default region')
        default_mask = ~regions_mask
        d_mults = self._default_mults

        if 'land_use' in d_mults:
            rlu = self._land_use[default_mask]
            lu_mult = self._create_land_use_mult(rlu, d_mults['land_use'])
            mults_arr[default_mask] = lu_mult

        if 'slope' in d_mults:
            r_slope = self._slope[default_mask]
            slope_mult = self._create_slope_mult(r_slope, d_mults['slope'])
            mults_arr[default_mask] = mults_arr[default_mask] * slope_mult

        # Set water multiplier last so we don't get super high multipliers at
        # water body boundaries next to steep slopes
        mults_arr[self._land_use == WATER_NLCD_CODE] = WATER_MULT

        return mults_arr

    def create_base_line_costs(self, capacity):
        """
        Get base line cost per cell raster for a given voltage

        Parameters
        ----------
        capacity : int
            Desired line capacity (MW). This must be in base_line_costs.

        Returns
        -------
        base_cost : numpy.ndarray
            Cost per cell raster in same shape as iso_regions
        """
        base_cost = np.full(self._iso_regions.shape, float('inf'),
                            dtype=np.float32)

        for iso in self._base_line_costs:
            logger.info(f'Processing costs for {iso} for {capacity}MW')
            iso_code = self._iso_lookup[iso]
            cost_per_mile = self._base_line_costs[iso][str(capacity)]
            cost_per_cell = cost_per_mile/METERS_IN_MILE*self._cell_size
            logger.debug(f'$/mile is {cost_per_mile}, $/cell is '
                         f'{cost_per_cell}')
            mask = self._iso_regions == iso_code
            base_cost[mask] = cost_per_cell

        return base_cost

    def _create_slope_mult(self, slope, config=None):
        """
        Create slope multiplier raster for a region. Unspecified slopes are
        left at 1.0

        Parameters
        ----------
        slope : numpy.ndarray
            Slope raster clipped to a region- "Terrain slope in % of grade"
        config : dict
            Multipliers and slope cut offs for hilly and mountain terrain
        config['hill_mult'] : float
            Multiplier for hilly terrain
        config['mtn_slope'] : float
            Multiplier for mountainous terrain
        config['hill_slope'] : float
            Slope at and above which a cell is classified as hilly.
        config['mtn_slope'] : float
            Slope at and above which a cell is classified as mountainous

        Returns
        -------
        numpy.ndarray
            Slope multiplier raster. Minimum value for any cell is 1.
        """
        if config is None:
            config = {}
        hill_mult = config.get('hill_mult', HILL_MULT)
        mtn_mult = config.get('mtn_mult', MTN_MULT)
        hill_slope = config.get('hill_slope', HILL_SLOPE)
        mtn_slope = config.get('mtn_slope', MTN_SLOPE)

        hilly = (slope >= hill_slope) & (slope < mtn_slope)
        mountainous = slope >= mtn_slope

        mult_raster = np.ones(slope.shape, dtype=np.float32)
        mult_raster[hilly] = hill_mult
        mult_raster[mountainous] = mtn_mult

        return mult_raster

    def _create_land_use_mult(self, land_use, multipliers):
        """
        Convert NLCD raster to land use multiplier raster for a single region.
        Land classes without specified multipliers are left a 1.

        Parameters
        ----------
        land_use : numpy.ndarray
           NLCD land user raster clipped to a region
        multipliers : dict
            Multiplier for for land classes, E.g. {'forest': 1.5}

        Returns
        -------
        numpy.ndarray
            Land use multiplier raster. Minimum value for any cell is 1.
        """
        mult_raster = np.ones(land_use.shape, dtype=np.float32)

        indices = []  # [(index0, multiplier0, _class_value0), ...]
        for _class, multiplier in multipliers.items():
            if _class not in self._land_use_classes:
                msg = f'Class {_class} not in self._land_use_classes: ' +\
                      f'{self._land_use_classes}'
                raise ValueError(msg)

            values = self._land_use_classes[_class]
            assert isinstance(values, list)

            for value in values:
                index = land_use == value
                indices.append((index, multiplier, value))

        for i in indices:
            mult_raster[i[0]] = i[1]

        return mult_raster
