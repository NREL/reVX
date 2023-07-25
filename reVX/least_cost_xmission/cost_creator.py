# -*- coding: utf-8 -*-
"""
Module to build and save least cost raster layers
"""
import logging
import numpy as np
import os

from reV.handlers.exclusions import ExclusionLayers

from reVX.handlers.geotiff import Geotiff
from reVX.utilities import ExclusionsConverter
from reVX.least_cost_xmission.config import XmissionConfig

logger = logging.getLogger(__name__)

NLCD_LAND_USE_CLASSES = {
    'cropland': [80, 81],
    'forest': [41, 42, 43],
    'wetland': [90, 95],
    'suburban': [21, 22, 23],
    'urban': [24],
}


class XmissionCostCreator(ExclusionsConverter):
    """
    Class to create and save Transmission cost layers to a .h5 Exclusion file
    - dist_to_coast (for valid sc_points)
    - base_costs
    - multiplier_*mw
    - xmission_barrier
    """
    PIXEL_SIZE = 90

    def __init__(self, h5_fpath, iso_regions_fpath, iso_lookup=None):
        """
        Parameters
        ----------
        h5_fpath : str
            H5 file to save costs to
        iso_regions_fpath : str
            ISO regions raster file
        iso_lookup : dict, optional
            Keys are ISO names used in various dicts with keys being int codes
            use in the iso_regions_fpath, by default None
        """
        super().__init__(h5_fpath)

        logger.debug('Loading ISO regions')
        with Geotiff(iso_regions_fpath) as gt:
            self._iso_regions = gt.values[0].astype('uint16')
            self._profile = gt.profile

        self._profile['dtype'] = 'float32'
        self._profile['nodata'] = None

        self._iso_lookup = iso_lookup if iso_lookup is not None else {}

        if not os.path.exists(self._excl_h5):
            self._init_h5(self._excl_h5, iso_regions_fpath)

        if 'ISO_regions' not in self.layers:
            self._geotiff_to_h5(self._excl_h5, 'ISO_regions',
                                iso_regions_fpath)
        else:
            self._check_geotiff(self._excl_h5, iso_regions_fpath)

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

        hill_mult = config.get('hill_mult', 1)
        mtn_mult = config.get('mtn_mult', 1)
        hill_slope = config.get('hill_slope', 2)
        mtn_slope = config.get('mtn_slope', 8)

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
                msg = ('Class {} not in land_use_classes: {}'
                       .format(_class, land_use_classes))
                logger.error(msg)
                raise ValueError(msg)

            values = land_use_classes[_class]
            assert isinstance(values, list)

            for value in values:
                index = land_use == value
                indices.append((index, multiplier, value))

        for i in indices:
            mult_raster[i[0]] = i[1]

        return mult_raster

    def compute_multipliers(self, iso_mults, excl_h5=None,
                            slope_layer='srtm_slope',
                            nlcd_layer='usa_mrlc_nlcd2011',
                            land_use_classes=None,
                            default_mults=None):
        """
        Create costs multiplier raster

        Parameters
        ----------
        iso_mults : dict
            Land use and slope multipliers for ISOs
        excl_h5 : str, optional
            Path to exclusion .h5 file containing NLCD and slope layers,
            if None assume NLCD and slope layers are in self._excl_h5,
            by default None
        sloper_layer : str, optional
            Name of slope layer in excl_h5, by default 'srtm_slope'
        ncld_layer : str, optional
            Name of NLCD (land use) layer in excl_h5, by default
            'usa_mrlc_nlcd2011'
        land_use_classes : dict, optional
            NCLD land use codes corresponding to use classes for multipliers.
            if None us NLCD_LAND_USE_CLASSES. by default None
        default_mults : dict, optional
            Multipliers for regions not specified in iso_mults_fpath.
            by default None

        Returns
        -------
        numpy.ndarray
            Costs multiplier raster, including slope and land use
        """
        if excl_h5 is None:
            excl_h5 = self._excl_h5

        logger.debug('Loading slope and land use rasters')
        with ExclusionLayers(excl_h5) as el:
            slope = el[slope_layer]
            land_use = el[nlcd_layer]

        logger.debug('Loading complete')

        assert self._iso_regions.shape == land_use.shape == slope.shape, \
            'All rasters must be the same shape'

        mults_arr = np.ones(self._iso_regions.shape, dtype=np.float32)
        regions_mask = np.full(mults_arr.shape, False, dtype=bool)

        if land_use_classes is None:
            land_use_classes = NLCD_LAND_USE_CLASSES

        for r_conf in iso_mults:
            iso = r_conf['iso']
            logger.info('Processing multipliers for region {}'.format(iso))

            if self._iso_lookup is not None:
                iso_name = iso
                iso = self._iso_lookup[iso]
                logger.debug('{} has id {}'.format(iso_name, iso))

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

        if default_mults is None:
            default_mults = {}

        default_mask = ~regions_mask

        if 'land_use' in default_mults:
            rlu = land_use[default_mask]
            lu_mult = self._compute_land_use_mult(rlu,
                                                  default_mults['land_use'],
                                                  land_use_classes)
            mults_arr[default_mask] = lu_mult

        if 'slope' in default_mults:
            r_slope = slope[default_mask]
            slope_mult = self._compute_slope_mult(r_slope,
                                                  default_mults['slope'])
            mults_arr[default_mask] = mults_arr[default_mask] * slope_mult

        # Set water multiplier last so we don't get super high multipliers at
        # water body boundaries next to steep slopes
        mults_arr[land_use == 11] = 10

        return mults_arr

    def compute_base_line_costs(self, capacity, base_line_costs):
        """
        Get base line cost per cell raster for a given voltage

        Parameters
        ----------
        capacity : int
            Desired line capacity (MW). This must be in base_line_costs.
        base_line_costs : dict
            Base line cost per mile for ISOs

        Returns
        -------
        base_cost : numpy.ndarray
            Cost per cell raster in same shape as iso_regions
        """
        base_cost = -1 * np.ones(self._iso_regions.shape, dtype=np.float32)

        for iso in base_line_costs:
            logger.info('Processing costs for {} for {}MW'
                        .format(iso, capacity))
            iso_code = self._iso_lookup[iso]
            cost_per_mile = base_line_costs[iso][str(capacity)]
            cost_per_cell = cost_per_mile / 1609.344 * self.PIXEL_SIZE
            logger.debug('$/mile is {}, $/cell is {}'
                         .format(cost_per_mile, cost_per_cell))
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
        logger.debug('Saving {} to h5'.format(layer_name))
        self._profile['nodata'] = None
        self._write_layer(self._excl_h5, layer_name, self._profile, data)

    def create_geotiff(self, geotiff_fpath, data):
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
        logger.debug('Saving {}'.format(geotiff_fpath))
        self.write_geotiff(geotiff_fpath, self._profile, data)

        # write_geotiff sets nodata to the max for the dtype. For float32 this
        # then breaks JSONifying the profile when writing h5s. Reset it.
        self._profile['nodata'] = nodata

    @classmethod
    def run(cls, h5_fpath, iso_regions_fpath, excl_h5=None,
            cost_configs=None, slope_layer='srtm_slope',
            nlcd_layer='usa_mrlc_nlcd2011', default_mults=None, tiff_dir=None,
            extra_layers=None):
        """
        Build cost rasters using base line costs and multipliers. Save to
        h5 file and tiff if desired

        Parameters
        ----------
        h5_fpath : str
            H5 file to save costs to
        iso_regions_fpath : str
            File with raster of ISO regions
        excl_h5 : str, optional
            Path to exclusion .h5 file containing NLCD and slope layers,
            if None assume NLCD and slope layers are in self._excl_h5,
            by default None
        cost_configs : str | dict, optional
            Path to json file containing Xmission cost configuration values,
            or jsonified dict of cost configuration values,
            or dictionary of configuration values,
            or dictionary of paths to config jsons,
            if None use defaults, by default None
        sloper_layer : str, optional
            Name of slope layer in excl_h5, by default 'srtm_slope'
        ncld_layer : str, optional
            Name of NLCD (land use) layer in excl_h5, by default
            'usa_mrlc_nlcd2011'
        default_mults : dict, optional
            Multipliers for regions not specified in iso_mults_fpath.
            by default None
        tiff_dir : str | None
            Path to save costs and intermediary rasters as geotiffs in. Don't
            save to geotiff if None.
        extra_layers : dict, optional
            Extra layers to add to h5 file, for example dist_to_coast. Format
            of dictionary is:
            {'layers': ['layers1', ...]
             'descriptions': optional,
             'scale_factors': optional,
             'kwargs': optional}
            by default None.
        """
        xc = XmissionConfig(config=cost_configs)

        xcc = cls(h5_fpath, iso_regions_fpath, iso_lookup=xc['iso_lookup'])

        if tiff_dir is None:
            save_geotiff = False
        else:
            save_geotiff = True

        mults_arr = xcc.compute_multipliers(
            xc['iso_multipliers'],
            excl_h5=excl_h5,
            slope_layer=slope_layer,
            nlcd_layer=nlcd_layer,
            land_use_classes=xc['land_use_classes'],
            default_mults=default_mults)

        xcc.save_layer('tie_line_multipliers', mults_arr)

        if save_geotiff:
            tiff_path = os.path.join(tiff_dir, 'multipliers.tif')
            xcc.create_geotiff(tiff_path, mults_arr)

        for power_class, capacity in xc['power_classes'].items():
            logger.info('Calculating costs for class {} using a {}MW line'
                        .format(power_class, capacity))
            blc_arr = xcc.compute_base_line_costs(capacity,
                                                  xc['base_line_costs'])

            if save_geotiff:
                tiff_path = os.path.join(tiff_dir,
                                         'base_line_costs_{}MW.tif'
                                         .format(capacity))
                xcc.create_geotiff(tiff_path, blc_arr)

            costs_arr = blc_arr * mults_arr

            xcc.save_layer('tie_line_costs_{}MW'.format(capacity), costs_arr)

            if save_geotiff:
                tiff_path = os.path.join(tiff_dir,
                                         'tie_line_costs{}MW.tif'
                                         .format(capacity))
                xcc.create_geotiff(tiff_path, costs_arr)

        if extra_layers:
            layers = extra_layers['layers']
            descriptions = extra_layers.get('descriptions', None)
            scale_factors = extra_layers.get('scale_factors', None)
            kwargs = extra_layers.get('kwargs', {})
            ExclusionsConverter.layers_to_h5(h5_fpath, layers,
                                             descriptions=descriptions,
                                             scale_factors=scale_factors,
                                             **kwargs)
