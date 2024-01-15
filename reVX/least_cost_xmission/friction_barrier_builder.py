"""
Build friction or barrier layers.
"""
from functools import reduce
import logging
from typing import Literal, Dict, TypedDict, Tuple, List

import numpy as np
import numpy.typing as npt

from reVX.least_cost_xmission.masks import Masks
from reVX.least_cost_xmission.trans_layer_io_handler import TransLayerIoHandler

logger = logging.getLogger(__name__)

# Terms for specifying masks. 'wet+' and 'dry+' indicated 'wet' + 'landfall'
# and 'dry' + 'landfall', respectively.
Extents = Literal['all', 'wet', 'wet+', 'landfall', 'dry+', 'dry']

class LayerConfig(TypedDict, total=False):
    """
    Friction and barrier layers config dict. 'extent' is mandatory. 'map' and
    'range' are exclusive, but one must be specified. 'value' must be specified
    if 'range' is used. Example configs can be seen in
    test_xmission_barrier_friction_builder.py in the tests directory.
    """
    extent: Extents  # extent to apply map or range to

    # Dict of values in raster (keys) and values to use for barrier/friction
    map: Dict[float, float]

    # Range of raster values to apply to barrier/friction. For value is minimum
    # value of range (inclusive). Second value is maximum value of range
    # (exclusive).
    range: Tuple[float, float]
    # Value to assign for barrier/friction for raster values within 'range'.
    value: float

    # If 'forced_inclusion' is specified, any cells with a value > 0 will
    # force the final value of corresponding cells to 0. Multiple forced
    # inclusions are allowed. This field is optional, and defaults to 'normal'
    # _type: Literal['normal', 'forced_inclusion']
    # TODO - The JSON parser should allow "type" as well as "_type"

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
                 dtype: npt.DTypeLike = 'float32'):
        """TODO

        """
        if _type not in ['friction', 'barrier']:
            raise AttributeError(
                f'_type must be "friction" or "barrier", received {_type}'
            )
        self._type = _type
        self._io_handler = io_handler
        self._masks = masks
        self._dtype = dtype

    def build_layer(self, layers: Dict[str, LayerConfig], save_tiff=True):
        """
        Combine multiple GeoTIFFs to create a friction or barrier layer, save
        to H5, and (optionally) as a GeoTIFF.

        Parameters
        ----------
        layers
            Dict of LayerConfigs keyed by GeoTIFF filenames.
        """
        logger.debug(f'Combining {self._type} layers')
        layer_arrays: List[Tuple[npt.NDArray, LayerConfig]] = []

        for fname, config in layers.items():
            data = self._io_handler.load_tiff(fname)
            layer_arrays.append((data, config))

        result = self._combine_layers(layer_arrays)

        if save_tiff:
            fname = BARRIER_TIFF_NAME if self._type == 'barrier' else \
                FRICTION_TIFF_NAME
            logger.debug(f'Writing combined {self._type} layers to {fname}')
            self._io_handler.save_tiff(result, fname)

        h5_layer_name = BARRIER_H5_LAYER_NAME if self._type == 'barrier' else \
            FRICTION_H5_LAYER_NAME
        logger.debug(f'Writing combined {self._type} layers to H5')
        self._io_handler.write_to_h5(result, h5_layer_name)

    def _combine_layers(self, layers: List[Tuple[npt.NDArray, LayerConfig]]
                       ) -> npt.NDArray:
        """
        Combine multiple arrays based on LayerConfigs and return composite
        array.

        Parameters
        ----------
        layers
            List of tuples, where the first value of each tuple is an array,
            and the second value is its LayerConfig.

        Returns
        -------
            The composite array.
        """
        # TODO - allow forced inclusions
        results: List[npt.NDArray] = []
        for data, config in layers:
            result = self._process_layer(data, config)
            results.append(result)
        return reduce(lambda a, b: a + b, results)

    def _process_layer(self, data: npt.NDArray, config: LayerConfig
                      ) -> npt.NDArray:
        """
        Process array using LayerConfig to create the desired layer. Desired
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
        self.__check_layer_config(config)

        if config['extent'] == 'wet':
            mask = self._masks.wet_mask
        elif config['extent'] == 'wet+':
            mask = self._masks.wet_plus_mask
        elif config['extent'] == 'dry':
            mask = self._masks.dry_mask
        elif config['extent'] == 'dry+':
            mask = self._masks.dry_plus_mask
        elif config['extent'] == 'landfall':
            mask = self._masks.landfall_mask
        else:
            mask = np.full(self._io_handler.shape, True, dtype=bool)

        processed = np.zeros(self._io_handler.shape, dtype=self._dtype)

        if 'range' in config:
            min, max = config['range']
            temp = np.where(
                np.logical_and(data >= min, data < max), config['value'], 0
            )
            processed[mask] = temp[mask]

        if 'map' in config:
            temp = np.zeros(self._io_handler.shape, dtype=self._dtype)
            for key, val in config['map'].items():
                temp[data == key] = val
            processed[mask] = temp[mask]

        return processed

    @staticmethod
    def __check_layer_config(config: LayerConfig):
        """
        Check if a LayerConfig is valid.

        Parameters
        ----------
        config
            The config dict to check

        Raises
        ------
        ValueError
            If an issue is detected
        """
        # TODO do all this with pydantic https://docs.pydantic.dev/latest/

        for key in config:
            if key not in ['extent', 'map', 'range', 'value']:
                raise ValueError(
                    f'Unknown key "{key}" found in config: {config}'
                )

        if 'extent' not in config:
            raise ValueError(
                f'Required key "extent" is missing from config {config}.'
            )

        if config['extent'] not in ['all', 'wet', 'wet+', 'landfall', 'dry+',
                                    'dry']:
            raise ValueError(
                f'Extent \'{config["extent"]}\' in config {config} is not '
                'supported.'
            )

        if 'map' in config and 'range' in config:
            raise ValueError(
                'Keys "map" and "range" are exclusionary but both were found '
                f'in config {config}'
            )

        if 'map' not in config and 'range' not in config:
            raise ValueError(
                'Either "map" or "range" must be specified, but neither was '
                f'found in config {config}'
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

        if 'map' in config and not isinstance(config['map'], dict):
            raise ValueError(f'"map" in config {config} must be a dict.')
