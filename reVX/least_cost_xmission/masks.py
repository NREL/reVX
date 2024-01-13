"""
Create, load, and store masks to determine land and sea.
"""
import os
import logging
from typing import Optional

import numpy as np
import numpy.typing as npt

from .utils import rasterize
from .trans_layer_io_handler import TransLayerIoHandler

logger = logging.getLogger(__name__)

# Mask array
Mask = npt.NDArray[np.bool_]

MASK_MSG = \
    'No mask available. Please run create_masks() or load_masks() first.'

class Masks:
    """
    Create, load, and store masks to determine land and sea.
    """
    LANDFALL_MASK_FNAME = 'landfall_mask.tif'  # One pixel width line at shore
    RAW_LAND_MASK_FNAME = 'raw_land_mask.tif'  # Rasterized land vector
    LAND_MASK_FNAME = 'land_mask.tif'  # = Raw mask - landfall mask
    OFFSHORE_MASK_FNAME = 'offshore_mask.tif'

    def __init__(self, io_handler: TransLayerIoHandler, masks_dir='.'):
        """ TODO

        Parameters
        ----------
        io_handler
            _description_
        masks_dir, optional
            _description_, by default '.'
        """
        self._io_handler = io_handler
        self._masks_dir = masks_dir
        os.makedirs(masks_dir, exist_ok=True)

        self._landfall_mask: Optional[Mask] = None
        self._dry_mask: Optional[Mask] = None
        self._wet_mask: Optional[Mask] = None

    @property
    def landfall_mask(self) -> Mask:
        """ Landfalls cells mask, only one cell wide """
        if self._landfall_mask is None:
            raise ValueError(MASK_MSG)
        return self._landfall_mask

    @property
    def wet_mask(self) -> Mask:
        """ Wet cells mask, does not include landfall cells """
        if self._wet_mask is None:
            raise ValueError(MASK_MSG)
        return self._wet_mask

    @property
    def dry_mask(self) -> Mask:
        """ Dry cells mask, does not include landfall cells """
        if self._dry_mask is None:
            raise ValueError(MASK_MSG)
        return self._dry_mask

    def create_masks(self, land_mask_shp_f: str, save_tiff: bool = False,
                     reproject_vector: bool = True):
        """
        Create the offshore and land mask layers from a polygon land vector
        file.

        Parameters
        ----------
        mask_shp_f
            Full path to land polygon gpgk or shp file
        save_tiff
            Save mask as tiff if true
        reproject_vector
            Reproject CRS of vector to match template raster if True.
        """
        logger.debug('Creating masks from %s', land_mask_shp_f)

        # Raw land is all land cells, include landfall cells
        raw_land = rasterize(land_mask_shp_f, self._io_handler.profile,
                             all_touched=True,
                             reproject_vector=reproject_vector)

        raw_land_mask: Mask = raw_land == 1

        # Offshore mask is inversion of raw land mask
        self._wet_mask = ~raw_land_mask

        landfall = rasterize(land_mask_shp_f, self._io_handler.profile,
                             reproject_vector=reproject_vector,
                             all_touched=True, boundary_only=True)
        self._landfall_mask = landfall == 1

        # XOR landfall and raw land to get all land cells, except landfall
        # cells
        self._dry_mask = np.logical_xor(self.landfall_mask,
                                         raw_land_mask)

        if save_tiff:
            logger.debug('Saving land and offshore masks to GeoTIFF')
            self.__save_mask(raw_land_mask, self.RAW_LAND_MASK_FNAME)
            self.__save_mask(self.wet_mask, self.OFFSHORE_MASK_FNAME)
            self.__save_mask(self.dry_mask, self.LAND_MASK_FNAME)
            self.__save_mask(self.landfall_mask, self.LANDFALL_MASK_FNAME)

    def load_masks(self):
        """
        Load the mask layers from GeoTIFFs. This does not need to be called if
        self.create_masks() was run previously. Mask files must be in the
        current directory.
        """

        self._dry_mask = self.__load_mask(self.LAND_MASK_FNAME)
        self._wet_mask = self.__load_mask(self.OFFSHORE_MASK_FNAME)
        self._landfall_mask = self.__load_mask(self.LANDFALL_MASK_FNAME)

        logger.info('Successfully loaded wet and dry masks')

    def __save_mask(self, data: npt.NDArray, fname: str):
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

    def __load_mask(self, fname: str) -> npt.NDArray[np.bool_]:
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

        raster = self._io_handler.load_tiff(full_fname)

        assert raster.max() == 1
        assert raster.min() == 0

        return raster == 1
