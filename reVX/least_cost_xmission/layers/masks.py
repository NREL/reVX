"""
Create, load, and store masks to determine land and sea.
"""
import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

from reVX.least_cost_xmission.layers.utils import rasterize
from reVX.least_cost_xmission.layers.transmission_layer_io_handler import (
    TransLayerIoHandler
)

logger = logging.getLogger(__name__)

# Mask array
MaskArr = npt.NDArray[np.bool_]

MASK_MSG = ('No mask available. Please run create_masks() or load_masks() '
            'first.')


class Masks:
    """
    Create, load, and store masks to determine land and sea.
    """
    LANDFALL_MASK_FNAME = 'landfall_mask.tif'
    """One pixel width line at shore"""
    RAW_LAND_MASK_FNAME = 'raw_land_mask.tif'
    """Rasterized land vector"""
    LAND_MASK_FNAME = 'land_mask.tif'
    """Raw mask - landfall mask"""
    OFFSHORE_MASK_FNAME = 'offshore_mask.tif'
    """Offshore mask filename"""

    def __init__(self, io_handler: TransLayerIoHandler, masks_dir='.'):
        """

        Parameters
        ----------
        io_handler : TransLayerIoHandler
            Transmission IO handler
        masks_dir : path-like, optional
            Directory for storing/finding mask GeoTIFFs. By default, ``'.'``.
        """
        self._io_handler = io_handler
        self._masks_dir = masks_dir
        os.makedirs(masks_dir, exist_ok=True)

        self._landfall_mask: Optional[MaskArr] = None
        self._dry_mask: Optional[MaskArr] = None
        self._wet_mask: Optional[MaskArr] = None
        self._dry_plus_mask: Optional[MaskArr] = None
        self._wet_plus_mask: Optional[MaskArr] = None

    @property
    def landfall_mask(self) -> MaskArr:
        """MaskArr: Landfalls cells mask, only one cell wide """
        if self._landfall_mask is None:
            raise ValueError(MASK_MSG)
        return self._landfall_mask

    @property
    def wet_mask(self) -> MaskArr:
        """MaskArr: Wet cells mask, does not include landfall cells """
        if self._wet_mask is None:
            raise ValueError(MASK_MSG)
        return self._wet_mask

    @property
    def dry_mask(self) -> MaskArr:
        """MaskArr: Dry cells mask, does not include landfall cells """
        if self._dry_mask is None:
            raise ValueError(MASK_MSG)
        return self._dry_mask

    @property
    def dry_plus_mask(self) -> MaskArr:
        """MaskArr: Dry cells mask, includes landfall cells """
        if self._dry_plus_mask is None:
            self._dry_plus_mask = np.logical_or(self.dry_mask,
                                                self.landfall_mask)
        return self._dry_plus_mask

    @property
    def wet_plus_mask(self) -> MaskArr:
        """MaskArr: Wet cells mask, includes landfall cells """
        if self._wet_plus_mask is None:
            self._wet_plus_mask = np.logical_or(self.wet_mask,
                                                self.landfall_mask)
        return self._wet_plus_mask

    def create_masks(self, land_mask_shp_f: str, save_tiff: bool = True,
                     reproject_vector: bool = True):
        """
        Create the offshore and land mask layers from a polygon land vector
        file.

        Parameters
        ----------
        mask_shp_f : str
            Full path to land polygon gpgk or shp file
        save_tiff : bool, optional
            Save mask as tiff if true. By default, ``True``.
        reproject_vector : bool, optional
            Reproject CRS of vector to match template raster if True.
            By default, ``True``.
        """
        logger.debug('Creating masks from %s', land_mask_shp_f)

        # Raw land is all land cells, include landfall cells
        raw_land = rasterize(land_mask_shp_f, self._io_handler.profile,
                             all_touched=True,
                             reproject_vector=reproject_vector, dtype='uint8')

        raw_land_mask: MaskArr = raw_land == 1

        # Offshore mask is inversion of raw land mask
        self._wet_mask = ~raw_land_mask

        landfall = rasterize(land_mask_shp_f, self._io_handler.profile,
                             reproject_vector=reproject_vector,
                             all_touched=True, boundary_only=True,
                             dtype='uint8')
        self._landfall_mask = landfall == 1

        # XOR landfall and raw land to get all land cells, except landfall
        # cells
        self._dry_mask = np.logical_xor(self.landfall_mask,
                                        raw_land_mask)

        logger.debug('Created all masks')

        if save_tiff:
            logger.debug('Saving masks to GeoTIFF')
            self._save_mask(raw_land_mask, self.RAW_LAND_MASK_FNAME)
            self._save_mask(self.wet_mask, self.OFFSHORE_MASK_FNAME)
            self._save_mask(self.dry_mask, self.LAND_MASK_FNAME)
            self._save_mask(self.landfall_mask, self.LANDFALL_MASK_FNAME)
            logger.debug('Completed saving all masks')

    def load_masks(self):
        """
        Load the mask layers from GeoTIFFs. This does not need to be called if
        self.create_masks() was run previously. Mask files must be in the
        current directory.
        """
        logger.debug('Loading masks')
        self._dry_mask = self._load_mask(self.LAND_MASK_FNAME)
        self._wet_mask = self._load_mask(self.OFFSHORE_MASK_FNAME)
        self._landfall_mask = self._load_mask(self.LANDFALL_MASK_FNAME)
        logger.debug('Successfully loaded wet, dry, and landfall masks')

    def _save_mask(self, data: npt.NDArray, fname: str):
        """
        Save mask to GeoTiff

        Parameters
        ----------
        data
            Data to save in GeoTiff
        fname
            Name of file to save
        """
        full_fname = os.path.join(self._masks_dir, fname)
        self._io_handler.save_tiff(data, full_fname)

    def _load_mask(self, fname: str) -> npt.NDArray[np.bool_]:
        """
        Load mask from GeoTIFF with sanity checking

        Parameters
        ----------
        fname
            Filename to load mask from

        Returns
        -------
            Mask data
        """
        full_fname = os.path.join(self._masks_dir, fname)

        if not Path(full_fname).exists():
            raise FileNotFoundError(f'Mask file at {full_fname} not found. '
                                    'Please create masks first.')

        raster = self._io_handler.load_tiff(full_fname)

        if raster.max() != 1:
            msg = (f'Maximum value in mask file {fname} is {raster.max()} but'
                   ' should be 1. Mask file appears to be corrupt. Please '
                   'recreate it.')
            logger.error(msg)
            raise ValueError(msg)
        if raster.min() != 0:
            msg = (f'Minimum value in mask file {fname} is {raster.min()} but'
                   ' should be 0. Mask file appears to be corrupt. Please '
                   'recreate it.')
            logger.error(msg)
            raise ValueError(msg)

        return raster == 1
