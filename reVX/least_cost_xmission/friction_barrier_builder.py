"""
Build friction or barrier layers from raster and vector data.
"""
import logging
from pathlib import Path
from functools import reduce
from typing import Literal, Dict, Tuple, List
from typing_extensions import TypedDict, Required

import numpy as np
import numpy.typing as npt

from reVX.least_cost_xmission.utils import rasterize
from reVX.least_cost_xmission.masks import MaskArr, Masks
from reVX.least_cost_xmission.config.constants import DEFAULT_DTYPE
from reVX.least_cost_xmission.transmission_layer_io_handler import TransLayerIoHandler

logger = logging.getLogger(__name__)

# Terms for specifying masks. 'wet+' and 'dry+' indicated 'wet' + 'landfall'
# and 'dry' + 'landfall', respectively.
Extents = Literal['all', 'wet', 'wet+', 'landfall', 'dry+', 'dry']

class Rasterize(TypedDict, total=False):
    """
    Rasterize a vector layer and apply a value to it.
    """
    value: Required[float]  # Value to burn in to raster
    buffer: float  # Optional value to buffer vector by (can be negative)
    reproject: bool  # Reproject vector to raster CRS if True

class FBLayerConfig(TypedDict, total=False):
    """
    Friction and barrier layers config dict. 'extent' is mandatory. 'map',
    'range', and 'rasterize' are exclusive, but one must be specified. 'value'
    must be specified if 'range' is used. Example configs can be seen in
    test_xmission_barrier_friction_builder.py in the tests directory.
    """
    extent: Required[Extents]  # extent to apply map or range to

    # Dict of values in raster (keys) and values to use for barrier/friction
    map: Dict[float, float]

    # Range of raster values to apply to barrier/friction. For value is minimum
    # value of range (inclusive). Second value is maximum value of range
    # (exclusive).
    range: Tuple[float, float]
    # Value to assign for barrier/friction for raster values within 'range'.
    value: float

    # Rasterize a vector and use as a friction or barrier layer
    rasterize: Rasterize

    # If 'forced_inclusion' is specified, any cells with a value > 0 will
    # force the final value of corresponding cells to 0. Multiple forced
    # inclusions are allowed. This field is optional, and defaults to 'normal'
    # type: Literal['normal', 'forced_inclusion']


BARRIER_H5_LAYER_NAME = 'transmission_barrier'
FRICTION_H5_LAYER_NAME = 'transmission_friction'

BARRIER_TIFF_NAME = 'barrier.tif'
FRICTION_TIFF_NAME = 'friction.tif'


class FrictionBarrierBuilder:
    """
    Build friction or barrier layers.
    """
    def __init__(self, _type: Literal['friction', 'barrier'],
                 io_handler: TransLayerIoHandler, masks: Masks,
                 dtype: npt.DTypeLike = DEFAULT_DTYPE):
        """
        Parameters
        ----------
        _type
            Type of layer being built
        io_handler
            IO handler
        masks
            Mask Handler
        dtype, optional
            Data type for final dataset, by default 'float32'
        """
        self._type = _type
        self._io_handler = io_handler
        self._masks = masks
        self._dtype = dtype

    def build_layer(self, layers: Dict[str, FBLayerConfig], save_tiff=True):
        """
        Combine multiple GeoTIFFs and vectors to create a friction or barrier
        layer, save to H5, and (optionally) as a GeoTIFF.

        Parameters
        ----------
        layers
            Dict of FBLayerConfigs keyed by GeoTIFF/vector filenames.
        """
        logger.debug(f'Combining {self._type} layers')
        layer_arrays: List[npt.NDArray] = []

        for fname, config in layers.items():
            logger.debug(f'Processing {fname} with config {config}')
            if Path(fname).suffix.lower() in ['.tif', '.tiff']:
                data = self._io_handler.load_tiff(fname)
                result = self._process_layer(data, config)
                layer_arrays.append(result)
            elif Path(fname).suffix.lower() in ['.shp', '.gpkg']:
                result = self._process_vector_layer(fname, config)
                layer_arrays.append(result)
            else:
                raise ValueError(f'Unsupported file extension on {fname}')

        # TODO - allow forced_inclusions
        result = reduce(lambda a, b: a + b, layer_arrays)

        if save_tiff:
            fname = BARRIER_TIFF_NAME if self._type == 'barrier' else \
                FRICTION_TIFF_NAME
            logger.debug(f'Writing combined {self._type} layers to {fname}')
            self._io_handler.save_tiff(result, fname)

        h5_layer_name = BARRIER_H5_LAYER_NAME if self._type == 'barrier' else \
            FRICTION_H5_LAYER_NAME
        logger.debug(f'Writing combined {self._type} layers to H5')
        self._io_handler.write_to_h5(result, h5_layer_name)

    def _process_layer(self, data: npt.NDArray, config: FBLayerConfig
                      ) -> npt.NDArray:
        """
        Process array using FBLayerConfig to create the desired layer. Desired
        "range" or "map" operation is only applied to the area indicated
        by the "extent".

        Parameters
        ----------
        data
            Array of data to process.
        config
            Definition of layer processing.

        Returns
        -------
            Transformed data.
        """
        self.__check_tiff_layer_config(config)
        mask = self.__get_mask(config['extent'])
        processed = np.zeros(self._io_handler.shape, dtype=self._dtype)

        # Assign all cells in a range to a value
        if 'range' in config:
            min, max = config['range']
            temp = np.where(
                np.logical_and(data >= min, data < max), config['value'], 0
            )
            processed[mask] = temp[mask]

        # Assign cells values based on map
        if 'map' in config:
            temp = np.zeros(self._io_handler.shape, dtype=self._dtype)
            for key, val in config['map'].items():
                temp[data == key] = val
            processed[mask] = temp[mask]

        return processed

    def _process_vector_layer(self, fname: str, config: FBLayerConfig
                             ) -> npt.NDArray:
        """
        Rasterize a vector layer

        Parameters
        ----------
        fname
            Name of vector layer to rasterize
        config
            Config for layer

        Returns
        -------
            Rasterized vector
        """
        if 'rasterize' not in config:
            raise ValueError(
                f'{fname} is a vector but the config is missing key '
                f'"rasterize": {config}'
            )

        mask = self.__get_mask(config['extent'])
        processed = np.zeros(self._io_handler.shape, dtype=self._dtype)

        r_config = config['rasterize']
        buffer = r_config['buffer'] if 'buffer' in r_config else None
        reproject = r_config['reproject'] if 'reproject' in r_config else True

        temp = rasterize(fname, self._io_handler.profile,
                         buffer_dist=buffer, burn_value=r_config['value'],
                         dtype=self._dtype, reproject_vector=reproject)
        processed[mask] = temp[mask]
        return processed

    def __get_mask(self, extent: Extents) -> MaskArr:
        """
        Get mask by requested extent

        Parameters
        ----------
        extent
            Extent of desired mask

        Returns
        -------
            Mask array
        """
        if extent == 'wet':
            mask = self._masks.wet_mask
        elif extent == 'wet+':
            mask = self._masks.wet_plus_mask
        elif extent == 'dry':
            mask = self._masks.dry_mask
        elif extent == 'dry+':
            mask = self._masks.dry_plus_mask
        elif extent == 'landfall':
            mask = self._masks.landfall_mask
        else:
            mask = np.full(self._io_handler.shape, True, dtype=bool)
        return mask

    @staticmethod
    def __check_tiff_layer_config(config: FBLayerConfig):
        """
        Check if a FBLayerConfig is valid for a GeoTIFF.

        Parameters
        ----------
        config
            The config dict to check

        Raises
        ------
        ValueError
            If an issue is detected
        """
        if 'rasterize' in config:
            raise ValueError(
                f'"rasterize" is only for vectors. Found in config {config}'
            )

        if 'map' in config and 'range' in config:
            raise ValueError(
                'Keys "map" and "range" are exclusionary but '
                'more than one was found in config {config}'
            )

        if 'map' not in config and 'range' not in config:
            raise ValueError(
                'Either "map" or "range" must be specified, but '
                f'neither were found in config {config}'
            )

        if 'range' in config and 'value' not in config:
            raise ValueError(
                '"value" must be specified if "range" is defined, but is '
                f'missing in config {config}'
            )

        if 'range' not in config and 'value' in config:
            raise ValueError(
                '"value" is specified but "range" is missing '
                f'in config {config}'
            )
