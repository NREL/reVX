"""
Build friction or barrier layers from raster and vector data.
"""
import logging
from pathlib import Path
from typing import Literal, Dict, Tuple, List
from typing_extensions import TypedDict, Required

import numpy as np
import numpy.typing as npt

from reVX.least_cost_xmission.layers.utils import rasterize
from reVX.least_cost_xmission.layers.masks import MaskArr, Masks
from reVX.least_cost_xmission.config.constants import DEFAULT_DTYPE
from reVX.least_cost_xmission.layers.transmission_layer_io_handler import \
    TransLayerIoHandler
from reVX.least_cost_xmission.config.constants import BARRIER_H5_LAYER_NAME, \
    BARRIER_TIFF, FRICTION_H5_LAYER_NAME, FRICTION_TIFF

logger = logging.getLogger(__name__)

# Terms for specifying masks. 'wet+' and 'dry+' indicated 'wet' + 'landfall'
# and 'dry' + 'landfall', respectively.
Extents = Literal['all', 'wet', 'wet+', 'landfall', 'dry+', 'dry']
ALL = 'all'


class Range(TypedDict, total=True):
    """
    Define a range of values in a raster to assign as a friction or barrrier.
    First value of min_max is lowest value of range (inclusive), second value
    of min_max is highest value in range (exclusive). `value` is the value used
    as friction barrier for any cells in the raster that fall within the range.
    """
    min_max: Tuple[float, float]
    value: float


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
    'range', and 'rasterize' are exclusive, but one must be specified.  Example
    configs can be seen in test_xmission_barrier_friction_builder.py in the
    tests directory.
    """
    extent: Required[Extents]  # extent to apply map or range to

    # Dict of values in raster (keys) and values to use for barrier/friction
    map: Dict[float, float]

    # One or more ranges of raster values to apply to barrier/friction. The
    # value of overlapping ranges are added together.
    range: List[Range]

    # Rasterize a vector and use as a friction or barrier layer
    rasterize: Rasterize

    # If 'forced_inclusion' is specified, any cells with a value > 0 will
    # force the final value of corresponding cells to 0. Multiple forced
    # inclusions are allowed. This field is optional, and defaults to 'normal'
    # e.g., type : Literal['normal', 'forced_inclusion']


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
        result = np.zeros(self._io_handler.shape, dtype=DEFAULT_DTYPE)

        for fname, config in layers.items():
            logger.debug(f'Processing {fname} with config {config}')
            if Path(fname).suffix.lower() in ['.tif', '.tiff']:
                data = self._io_handler.load_tiff(fname, reproject=True)
                temp = self._process_raster_layer(data, config)
                result += temp
            elif Path(fname).suffix.lower() in ['.shp', '.gpkg']:
                temp = self._process_vector_layer(fname, config)
                result += temp
            else:
                raise ValueError(f'Unsupported file extension on {fname}')

        # TODO - allow forced_inclusions

        if save_tiff:
            fname = BARRIER_TIFF if self._type == 'barrier' else \
                FRICTION_TIFF
            logger.debug(f'Writing combined {self._type} layers to {fname}')
            self._io_handler.save_tiff(result, fname)

        h5_layer_name = BARRIER_H5_LAYER_NAME if self._type == 'barrier' else \
            FRICTION_H5_LAYER_NAME
        logger.debug(f'Writing combined {self._type} layers to H5')
        self._io_handler.write_to_h5(result, h5_layer_name)

    def _process_raster_layer(self, data: npt.NDArray, config: FBLayerConfig
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

        # Assign all cells one or more ranges to a value
        if 'range' in config:
            processed = np.zeros(self._io_handler.shape, dtype=self._dtype)
            if config['extent'] != 'all':
                mask = self.__get_mask(config['extent'])

            for range in config['range']:
                min, max = range['min_max']
                value = range['value']
                temp = np.where(
                    np.logical_and(data >= min, data < max), value, 0
                )

                if config['extent'] == ALL:
                    processed += temp
                    continue

                processed[mask] += temp[mask]

            return processed

        # Assign cells values based on map
        if 'map' in config:
            temp = np.zeros(self._io_handler.shape, dtype=self._dtype)
            for key, val in config['map'].items():
                temp[data == key] = val

            if config['extent'] == ALL:
                return temp

            mask = self.__get_mask(config['extent'])
            processed = np.zeros(self._io_handler.shape, dtype=self._dtype)
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

        r_config = config['rasterize']
        buffer = r_config['buffer'] if 'buffer' in r_config else None
        reproject = r_config['reproject'] if 'reproject' in r_config else True

        temp = rasterize(fname, self._io_handler.profile,
                         buffer_dist=buffer, burn_value=r_config['value'],
                         dtype=self._dtype, reproject_vector=reproject)

        if config['extent'] == ALL:
            return temp

        mask = self.__get_mask(config['extent'])
        processed = np.zeros(self._io_handler.shape, dtype=self._dtype)
        processed[mask] = temp[mask]
        return processed

    def __get_mask(self, extent: Extents) -> MaskArr:
        """
        Get mask by requested extent

        Parameters
        ----------
        extent
            Extent of desired mask, 'all' is not allowed.

        Returns
        -------
            Mask array
        """
        if extent == ALL:
            raise AttributeError(
                f'Mask for extent of {extent} is unnecessary'
            )

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
            raise AttributeError(f'Unknown mask type: {extent}')

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
