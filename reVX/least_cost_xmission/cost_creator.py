# -*- coding: utf-8 -*-
"""
Module to build and save least cost raster layers
"""
import logging
import numpy as np
import os

from reV.handlers.exclusions import ExclusionLayers

from reVX.handlers.geotiff import Geotiff
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
    def __init__(self, h5_fpath, iso_regions_fpath, iso_lookup):
        """
        Parameters
        ----------
        h5_fpath : str
            H5 file to save costs to
        iso_regions_fpath : str
            ISO regions raster file
        iso_lookup : dict
            Keys are ISO names used in various dicts with keys being int codes
            use in the iso_regions_fpath.
        """
        super().__init__(h5_fpath)

        logger.debug('Loading ISO regions')
        with Geotiff(iso_regions_fpath) as gt:
            self._iso_regions = gt.values[0]
            self._profile = gt.profile

        self._profile['dtype'] = 'float32'
        self._profile['nodata'] = None

        self._iso_lookup = iso_lookup

        if not os.path.exists(h5_fpath):
            self._init_h5(h5_fpath, iso_regions_fpath)

    @classmethod
    def build_cost_rasters(cls, h5_fpath, iso_regions_fpath, input_h5_fpath,
                           iso_mults_fpath=None, base_line_costs_fpath=None,
                           iso_lookup_fpath=None, power_classes_fpath=None,
                           slope_lyr='srtm_slope',
                           nlcd_lyr='usa_mrlc_nlcd2011',
                           land_use_classes=None,
                           default_mults=None, tiff_dir=None,
                           cell_size=CELL_SIZE,

                           # TODO - ditch this in a bit
                           save_layer=True,
                           ):
        """
        Build cost rasters using base line costs and multipliers. Save to
        h5 file or tiff

        Parameters
        ----------
        h5_fpath : str
            H5 file to save costs to
        iso_regions_fpath : str
            File with raster of ISO regions
        input_h5_fpath : str
            File with NLCD and slope layers
        iso_mults_fpath : None | str
            ISO multipliers JSON file. Use default if None
        base_line_costs_fpath : None | str
            Base line cost per mile for ISOs file. Use default if None
        iso_lookup_fpath : None | str
            JSON file linking ISO region code to name. Use default if None
        power_classes_fpath : None | str
            JSON of line capacities to use for reV power classes. Use default
            if None
        sloper_lyr : str
            Name of slope layer in input_h5_fpath
        ncld_lyr : str
            Name of NLCD (land use) layer in input_h5_fpath
        land_use_classes : dict | None
            NCLD land use codes corresponding to use classes for multipliers.
            If None use defaults.
        default_mults : None | dict
            Multipliers for regions not specified in iso_mults_fpath. Use
            defaults if None..
        tiff_dir : str | None
            Path to save costs and intermediary rasters as geotiffs in. Don't
            save to geotiff if None.
        cell_size : int
            Raster cell size for all layers (meters). Assumed to be square.
        """
        # TODO - verify profile for iso_regions_fpath and input_h5_fpath are
        # comparable.

        xc = XmissionConfig(iso_mults_fpath=iso_mults_fpath,
                            base_line_costs_fpath=base_line_costs_fpath,
                            iso_lookup_fpath=iso_lookup_fpath,
                            power_classes_fpath=power_classes_fpath)

        xcc = cls(h5_fpath, iso_regions_fpath, xc['iso_lookup'])

        if tiff_dir is None:
            save_geotiff = False
        else:
            save_geotiff = True

        if land_use_classes is None:
            land_use_classes = NLCD_LAND_USE_CLASSES

        mults_arr = xcc.compute_multipliers(input_h5_fpath, slope_lyr,
                                            nlcd_lyr, land_use_classes,
                                            xc['iso_mults'], default_mults)

        if save_geotiff:
            xcc.create_geotiff(tiff_dir, 'multipliers.tif', mults_arr)

        # TODO - remove
        xcc.save_layer('mults', mults_arr)

        for power_class, capacity in xc['power_classes'].items():
            logger.info(f'Calculating costs for class {power_class} using a '
                        f'{capacity}MW line')
            blc_arr = xcc.compute_base_line_costs(capacity,
                                                  xc['base_line_costs'],
                                                  cell_size)

            if save_geotiff:
                xcc.create_geotiff(tiff_dir, f'blc_{capacity}MW.tif', blc_arr)

            costs_arr = blc_arr * mults_arr

            if save_layer:
                xcc.save_layer(f'tie_line_costs_{capacity}MW', costs_arr)

            if save_geotiff:
                xcc.create_geotiff(tiff_dir, f'tie_line_costs{capacity}MW.tif',
                                   costs_arr)

    def compute_multipliers(self, input_h5_fpath, slope_lyr, nlcd_lyr,
                            land_use_classes, iso_mults, d_mults):
        """
        Create costs multiplier raster

        Parameters
        ----------
        input_h5_fpath : str
            File with NLCD and slope layers
        sloper_lyr : str
            Name of slope layer in input_h5_fpath
        ncld_lyr : str
            Name of NLCD (land use) layer in input_h5_fpath
        land_use_classes : dict
            NCLD land use codes corresponding to use classes for multipliers
        iso_mults : dict
            Land use and slope multipliers for ISOs
        d_mults : dict | None
            Default line cost multipliers for areas not covered by the ISO
            regions.

        Returns
        -------
        numpy.ndarray
            Costs multiplier raster, including slope and land use
        """
        logger.debug('Loading slope and land use rasters')
        with ExclusionLayers(input_h5_fpath) as el:
            slope = el[slope_lyr]
            land_use = el[nlcd_lyr]
        logger.debug('Loading complete')

        assert self._iso_regions.shape == land_use.shape == slope.shape, \
            'All rasters must be the same shape'

        mults_arr = np.ones(self._iso_regions.shape, dtype=np.float32)
        regions_mask = np.full(mults_arr.shape, False, dtype=bool)

        for r_conf in iso_mults:
            iso = r_conf['iso']
            logger.info(f'Processing multipliers for region {iso}')

            if self._iso_lookup is not None:
                iso_name = iso
                iso = self._iso_lookup[iso]
                logger.debug(f'{iso_name} has id {iso}')

            mask = self._iso_regions == iso
            regions_mask = regions_mask | mask

            if 'land_use' in r_conf:
                r_lu = land_use[mask]
                lu_mult = self._compute_land_use_mult(r_lu, r_conf['land_use'],
                                                      land_use_classes)
                mults_arr[mask] = lu_mult

            if 'slope' in r_conf:
                r_slope = slope[mask]
                slope_mult = self._compute_slope_mult(r_slope, r_conf['slope'])
                mults_arr[mask] = mults_arr[mask] * slope_mult

        # Calculate multipliers for regions not defined in `config`
        logger.debug('Processing default region')
        if d_mults is None:
            d_mults = {}
        default_mask = ~regions_mask

        if 'land_use' in d_mults:
            rlu = land_use[default_mask]
            lu_mult = self._compute_land_use_mult(rlu, d_mults['land_use'])
            mults_arr[default_mask] = lu_mult

        if 'slope' in d_mults:
            r_slope = slope[default_mask]
            slope_mult = self._compute_slope_mult(r_slope, d_mults['slope'])
            mults_arr[default_mask] = mults_arr[default_mask] * slope_mult

        # Set water multiplier last so we don't get super high multipliers at
        # water body boundaries next to steep slopes
        mults_arr[land_use == WATER_NLCD_CODE] = WATER_MULT

        return mults_arr

    def compute_base_line_costs(self, capacity, base_line_costs, cell_size):
        """
        Get base line cost per cell raster for a given voltage

        Parameters
        ----------
        capacity : int
            Desired line capacity (MW). This must be in base_line_costs.
        base_line_costs : dict
            Base line cost per mile for ISOs
        cell_size : int
            Raster cell size for all layers (meters). Assumed to be square.

        Returns
        -------
        base_cost : numpy.ndarray
            Cost per cell raster in same shape as iso_regions
        """
        base_cost = np.full(self._iso_regions.shape, float('inf'),
                            dtype=np.float32)

        for iso in base_line_costs:
            logger.info(f'Processing costs for {iso} for {capacity}MW')
            iso_code = self._iso_lookup[iso]
            cost_per_mile = base_line_costs[iso][str(capacity)]
            cost_per_cell = cost_per_mile / METERS_IN_MILE * cell_size
            logger.debug(f'$/mile is {cost_per_mile}, $/cell is '
                         f'{cost_per_cell}')
            mask = self._iso_regions == iso_code
            base_cost[mask] = cost_per_cell

        return base_cost

    def save_layer(self, layer_name, data):
        """
        Save data to h5 file

        Parameters
        ----------
        layer_name : str
            Name of layer to save in h5
        data : np.ndarray
            Data to savein h5
        """
        logger.debug(f'Saving {layer_name} to h5')
        self._profile['nodata'] = None
        self._write_layer(self._excl_h5, layer_name, self._profile, data)

    def create_geotiff(self, tiff_dir, geotiff_fpath, data):
        """
        Save geotiff to disk

        Parameters
        ----------
        tiff_dir : str
            Directory to save tiffs in
        geotiff_fpath : str
            File name of tiff including extension
        data : np.ndarray
            Data to save
        """
        nodata = self._profile['nodata']
        if tiff_dir != '':
            geotiff_fpath = os.path.join(tiff_dir, geotiff_fpath)
        logger.debug(f'Saving {geotiff_fpath}')
        self._write_geotiff(geotiff_fpath, self._profile, data)

        # _write_geotiff sets nodata to the max for the dtype. For float32 this
        # then breaks JSONifying the profile when writing h5s. Reset it.
        self._profile['nodata'] = nodata

    @staticmethod
    def _compute_slope_mult(slope, config=None):
        """
        Create slope multiplier raster for a region. Unspecified slopes are
        left at 1.0

        Parameters
        ----------
        slope : numpy.ndarray
            Slope raster clipped to a region- "Terrain slope in % of grade"
        config : dict | None
            Multipliers and slope cut offs for hilly and mountain terrain.
            Use defaults if None.
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

    @staticmethod
    def _compute_land_use_mult(land_use, multipliers, land_use_classes):
        """
        Convert NLCD raster to land use multiplier raster for a single region.
        Land classes without specified multipliers are left a 1.

        Parameters
        ----------
        land_use : numpy.ndarray
           NLCD land user raster clipped to a region
        multipliers : dict
            Multiplier for for land classes, E.g. {'forest': 1.5}
        land_use_classes : dict
            NCLD land use codes corresponding to use classes for multipliers

        Returns
        -------
        numpy.ndarray
            Land use multiplier raster. Minimum value for any cell is 1.
        """
        mult_raster = np.ones(land_use.shape, dtype=np.float32)

        indices = []  # [(index0, multiplier0, _class_value0), ...]
        for _class, multiplier in multipliers.items():
            if _class not in land_use_classes:
                msg = f'Class {_class} not in land_use_classes: ' +\
                      f'{land_use_classes}'
                raise ValueError(msg)

            values = land_use_classes[_class]
            assert isinstance(values, list)

            for value in values:
                index = land_use == value
                indices.append((index, multiplier, value))

        for i in indices:
            mult_raster[i[0]] = i[1]

        return mult_raster
