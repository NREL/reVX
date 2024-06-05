# -*- coding: utf-8 -*-
"""
Module to build and save dry (land) cost raster layers
"""
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, Union, List, Tuple

from reVX.handlers.layered_h5 import LayeredTransmissionH5
from reVX.least_cost_xmission.config import (XmissionConfig, LandUseClasses,
                                             LandUseMultipliers,
                                             SlopeMultipliers, IsoMultipliers)
from reVX.least_cost_xmission.config.constants import (CELL_SIZE,
                                                       WATER_MULT,
                                                       WATER_NLCD_CODE,
                                                       METERS_IN_MILE,
                                                       HILL_MULT, MTN_MULT,
                                                       HILL_SLOPE, MTN_SLOPE,
                                                       DEFAULT_DTYPE,
                                                       DRY_MULTIPLIER_TIFF)
from reVX.least_cost_xmission.layers.masks import MaskArr
from reVX.least_cost_xmission.layers.base import BaseLayerCreator

logger = logging.getLogger(__name__)


NLCD_LAND_USE_CLASSES: LandUseClasses = {
    'cropland': [80, 81],
    'forest': [41, 42, 43],
    'wetland': [90, 95],
    'suburban': [21, 22, 23],
    'urban': [24],
}


class DryCostCreator(BaseLayerCreator):
    """
    Class to create and save dry transmission cost layers
    """
    def __init__(self, io_handler: LayeredTransmissionH5,
                 mask, output_tiff_dir=".",
                 dtype: npt.DTypeLike = DEFAULT_DTYPE,
                 cell_size=CELL_SIZE):
        """
        Parameters
        ----------
        io_handler : :class:`LayeredTransmissionH5`
            Transmission layer IO handler
        mask : ndarray
            Array representing mask for layer values.
        output_tiff_dir : path-like, optional
            Directory where cost layers should be saved as GeoTIFF.
            By default, ``"."``.
        dtype : np.dtype, optional
            Data type for final dataset. By default, ``float32``.
        cell_size : int, optional
            Side length of each cell, in meters. Cells are assumed to be
            square. By default, :obj:`CELL_SIZE`.
        """
        super().__init__(io_handler=io_handler, mask=mask,
                         output_tiff_dir=output_tiff_dir, dtype=dtype,
                         cell_size=cell_size)
        self._iso_lookup: Dict[str, int] = {}

    def build(self, iso_region_tiff: str, nlcd_tiff: str, slope_tiff: str,
              cost_configs: Optional[Union[str, Dict]] = None,
              default_mults: Optional[IsoMultipliers] = None,
              extra_tiffs: Optional[List[str]] = None):
        """
        Build cost rasters using base line costs and multipliers. Save to
        GeoTIFF.

        Parameters
        ----------
        iso_region_tiff : str
            Path to the ISO region GeoTIFF
        nlcd_tiff : str
            Path to the National Land Coverage Database GeoTIFF
        slope_tiff : str
            Path to the slope GeoTIFF. Slope is in decimal percent.
        cost_configs : dict | str, optional
            Dictionary or path to json file contianing dictionary with
            Xmission cost configuration values. Valid configuration keys
            are:

                - "base_line_costs"
                - "iso_lookup"
                - "iso_multipliers"
                - "land_use_classes"
                - "new_substation_costs"
                - "power_classes"
                - "power_to_voltage"
                - "transformer_costs"
                - "upgrade_substation_costs"

            Each of these keys should point to a dictionary or a path to
            a separate json file contianing a dictionary of
            configurations for each section.
        default_mults : IsoMultipliers, optional
            Multipliers for regions not specified in iso_mults_fpath.
            by default None
        extra_tiffs : list, optional
            Optional list of extra GeoTIFFs to add to cost H5 file (e.g.
            a transmission barrier file). By default, ``None``, which
            does not add any extra layers.
        """
        xc = XmissionConfig(config=cost_configs)
        self._iso_lookup = xc['iso_lookup']

        logger.debug('Loading ISO region, slope and land use rasters')
        iso_layer = self._io_handler.load_data_using_h5_profile(
            iso_region_tiff, reproject=True)
        slope_layer = self._io_handler.load_data_using_h5_profile(
            slope_tiff, reproject=True)
        nlcd_layer = self._io_handler.load_data_using_h5_profile(
            nlcd_tiff, reproject=True)
        logger.debug('Loading complete')

        lu_classes = xc['land_use_classes']
        iso_mults: List[IsoMultipliers] = xc['iso_multipliers']
        mults_arr = self._compute_multipliers(iso_mults, iso_layer,
                                              slope_layer, nlcd_layer,
                                              land_use_classes=lu_classes,
                                              default_mults=default_mults)

        logger.debug('Saving multipliers array GeoTIFF')
        mult_tiff = self.output_tiff_dir / DRY_MULTIPLIER_TIFF
        self._io_handler.save_data_using_h5_profile(mults_arr, mult_tiff)

        if self._io_handler is not None:
            tiff_layers = [iso_region_tiff, slope_tiff, nlcd_tiff, mult_tiff]
            tiff_layers += extra_tiffs or []
            for layer_fp in tiff_layers:
                layer_name = Path(layer_fp).stem
                out = self._io_handler.load_data_using_h5_profile(
                    layer_fp, reproject=True)
                logger.debug(f'Writing {layer_name} to H5')
                self._io_handler.write_layer_to_h5(out, layer_name)

        for power_class, capacity in xc['power_classes'].items():
            logger.info('Calculating costs for class %s using a %sMW line',
                        power_class, capacity)
            blc_arr = self._compute_base_line_costs(capacity,
                                                    xc['base_line_costs'],
                                                    iso_layer)

            base_costs_tiff = 'base_line_costs_{}MW.tif'.format(capacity)
            out_fp = self.output_tiff_dir / base_costs_tiff
            self._io_handler.save_data_using_h5_profile(blc_arr, out_fp)

            # Calculate total costs w/ multipliers
            costs_arr = blc_arr * mults_arr

            dry_layer_name = 'tie_line_costs_{}MW'.format(capacity)
            tie_line_costs_tiff = '{}.tif'.format(dry_layer_name)
            out_fp = self.output_tiff_dir / tie_line_costs_tiff
            costs_arr[~self._mask] = 0
            self._io_handler.save_data_using_h5_profile(costs_arr, out_fp)
            if self._io_handler is not None:
                out = self._io_handler.load_data_using_h5_profile(
                    out_fp, reproject=True)
                logger.debug('Writing dry costs to H5')
                self._io_handler.write_layer_to_h5(out, dry_layer_name)

    @staticmethod
    def _compute_slope_mult(slope: npt.NDArray,
                            config: Optional[SlopeMultipliers] = None
                            ) -> npt.NDArray:
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
    def _compute_land_use_mult(land_use: npt.NDArray,
                               multipliers: LandUseMultipliers,
                               land_use_classes: LandUseClasses
                               ) -> npt.NDArray:
        """
        Convert NLCD raster to land use multiplier raster for a single ISO
        region. Land classes without specified multipliers are left a 1.

        Parameters
        ----------
        land_use : numpy.ndarray
           NLCD land user raster clipped to a region
        multipliers : LandUseMultipliers
            Multiplier for for land classes, E.g. {'forest': 1.5}
        land_use_classes : dict
            NCLD land use codes corresponding to use classes for multipliers

        Returns
        -------
        numpy.ndarray
            Land use multiplier raster. Minimum value for any cell is 1.
        """
        mult_raster = np.ones(land_use.shape, dtype=np.float32)

        # Determine mask arrays for NCLD values and multiplier to apply
        indices: List[Tuple[MaskArr, float, int]] = []
        multiplier: float
        for _class, multiplier in multipliers.items():  # type: ignore
            if _class not in land_use_classes:
                msg = ('Class {} not in land_use_classes: {}'
                       .format(_class, land_use_classes))
                logger.error(msg)
                raise ValueError(msg)

            nlcd_values: List[int] = land_use_classes[_class]  # type: ignore
            assert isinstance(nlcd_values, list)

            for nlcd_value in nlcd_values:
                index: MaskArr = land_use == nlcd_value
                indices.append((index, multiplier, nlcd_value))

        # Apply multipliers to appropriate cells
        for i in indices:
            mult_raster[i[0]] = i[1]

        return mult_raster

    def _compute_multipliers(self, iso_mults: List[IsoMultipliers],
                             iso_layer: npt.NDArray,
                             slope_layer: npt.NDArray,
                             land_use_layer: npt.NDArray,
                             land_use_classes: Optional[LandUseClasses] = None,
                             default_mults: Optional[IsoMultipliers] = None
                             ) -> npt.NDArray:
        """
        Create costs multiplier raster

        Parameters
        ----------
        iso_mults : List[IsoMultipliers]
            Land use and slope multipliers for ISOs
        sloper_layer : str, optional
            Name of slope layer in excl_h5, by default 'srtm_slope'
        ncld_layer : str, optional
            Name of NLCD (land use) layer in excl_h5, by default
            'usa_mrlc_nlcd2011'
        land_use_classes : dict, optional
            NCLD land use codes corresponding to use classes for multipliers.
            If None, use NLCD_LAND_USE_CLASSES. By default None
        default_mults : dict, optional
            Multipliers for regions not specified in iso_mults_fpath.
            By default None

        Returns
        -------
        numpy.ndarray
            Costs multiplier raster, including slope and land use
        """
        mults_arr = np.ones(self.shape, dtype=np.float32)
        regions_mask: MaskArr = np.full(self.shape, False, dtype=bool)

        if land_use_classes is None:
            land_use_classes = NLCD_LAND_USE_CLASSES

        for r_conf in iso_mults:
            iso_name = r_conf['iso']
            logger.info('Processing multipliers for region %s', iso_name)

            iso = self._iso_lookup[iso_name]
            logger.debug('ISO %s has id %s', iso_name, iso)

            mask: MaskArr = iso_layer == iso
            regions_mask = regions_mask | mask

            if 'land_use' in r_conf:
                r_lu = land_use_layer[mask]
                lu_mult = self._compute_land_use_mult(r_lu, r_conf['land_use'],
                                                      land_use_classes)
                mults_arr[mask] = lu_mult

            if 'slope' in r_conf:
                r_slope = slope_layer[mask]
                slope_mult = self._compute_slope_mult(r_slope, r_conf['slope'])
                mults_arr[mask] = mults_arr[mask] * slope_mult

        # Calculate multipliers for regions not defined in `config`
        logger.debug('Processing default region')
        if default_mults is not None:
            default_mask = ~regions_mask

            if 'land_use' in default_mults:
                region_land_use = land_use_layer[default_mask]
                lu_mult_dict = default_mults['land_use']
                lu_mult = self._compute_land_use_mult(region_land_use,
                                                      lu_mult_dict,
                                                      land_use_classes)
                mults_arr[default_mask] = lu_mult

            if 'slope' in default_mults:
                region_slope = slope_layer[default_mask]
                slope_mult = self._compute_slope_mult(region_slope,
                                                      default_mults['slope'])
                mults_arr[default_mask] = mults_arr[default_mask] * slope_mult

        # Set water multiplier last so we don't get super high multipliers at
        # water body boundaries next to steep slopes
        mults_arr[land_use_layer == WATER_NLCD_CODE] = WATER_MULT

        return mults_arr

    def _compute_base_line_costs(self, capacity: int,
                                 base_line_costs: Dict[str, Dict[str, float]],
                                 iso_layer: npt.NDArray) -> npt.NDArray:
        """
        Get base line cost per cell raster for a given voltage

        Parameters
        ----------
        capacity : int
            Desired line capacity (MW). This must be in base_line_costs.
        base_line_costs : dict
            Base line cost per mile for ISOs
        iso_layer: numpy.ndarray
            Raster defining ISO regions.

        Returns
        -------
        base_cost : numpy.ndarray
            Cost per cell raster in standard shape
        """
        base_cost = np.zeros(self.shape, dtype=np.float32)

        for iso in base_line_costs:
            logger.info('Processing costs for %s for %sMW', iso, capacity)
            iso_code = self._iso_lookup[iso]
            cost_per_mile = base_line_costs[iso][str(capacity)]
            cost_per_cell = cost_per_mile / METERS_IN_MILE * self._cell_size

            logger.debug('Base line $/mile is %s, $/cell is %s', cost_per_mile,
                         cost_per_cell)
            mask = iso_layer == iso_code
            base_cost[mask] = cost_per_cell

        return base_cost
