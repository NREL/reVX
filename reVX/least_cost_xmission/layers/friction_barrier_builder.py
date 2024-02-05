"""
Build friction or barrier layers from raster and vector data.
"""
import logging
from pathlib import Path
from typing import Literal, Dict

import numpy as np
import numpy.typing as npt
from reVX.config.transmission_layer_creation import Extents, FBLayerConfig

from reVX.least_cost_xmission.layers.utils import rasterize
from reVX.least_cost_xmission.layers.masks import MaskArr, Masks
from reVX.least_cost_xmission.config.constants import (DEFAULT_DTYPE,
                                                       RAW_BARRIER_TIFF,
                                                       FRICTION_TIFF)
from reVX.least_cost_xmission.layers.transmission_layer_io_handler import (
    TransLayerIoHandler
)

logger = logging.getLogger(__name__)

ALL = 'all'


class FrictionBarrierBuilder:
    """
    Build friction or barrier layers.
    """
    def __init__(self, type_: Literal['friction', 'barrier'],
                 io_handler: TransLayerIoHandler, masks: Masks,
                 dtype: npt.DTypeLike = DEFAULT_DTYPE):
        """
        Parameters
        ----------
        type_ : {'friction', 'barrier'}
            Type of layer being built
        io_handler : TransLayerIoHandler
            IO handler
        masks : Masks
            Mask Handler
        dtype : np.dtype, optional
            Data type for final dataset. By default, ``float32``.
        """
        self._type = type_
        self._io_handler = io_handler
        self._masks = masks
        self._dtype = dtype

    def build_layer(self, layers: Dict[str, FBLayerConfig]):
        """
        Combine multiple GeoTIFFs and vectors to create a friction or barrier
        layer and save to GeoTIFF.

        Parameters
        ----------
        layers : dict
            Dict of FBLayerConfigs keyed by GeoTIFF/vector filenames.
        """
        logger.debug(f'Combining {self._type} layers')
        result = np.zeros(self._io_handler.shape, dtype=DEFAULT_DTYPE)
        fi_layers: Dict[str, FBLayerConfig] = {}

        for fname, config in layers.items():
            if config.forced_inclusion:
                fi_layers[fname] = config
                continue

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

        result = self._process_forced_inclusions(result, fi_layers)

        fname = (RAW_BARRIER_TIFF
                 if self._type == 'barrier' else FRICTION_TIFF)
        logger.debug(f'Writing combined {self._type} layers to {fname}')
        self._io_handler.save_tiff(result, fname)

    def _process_raster_layer(self, data: npt.NDArray,  # type: ignore[return]
                              config: FBLayerConfig) -> npt.NDArray:
        """
        Process array using FBLayerConfig to create the desired layer. Desired
        "range" or "map" operation is only applied to the area indicated by the
        "extent".

        Parameters
        ----------
        data : array-like
            Array of data to process.
        config : FBLayerConfig
            Definition of layer processing.

        Returns
        -------
        array-like | None
            Transformed data.
        """
        self._check_tiff_layer_config(config)

        # Assign all cells one or more ranges to a value
        if config.range is not None:
            processed = np.zeros(self._io_handler.shape, dtype=self._dtype)
            if config.extent != ALL:
                mask = self._get_mask(config.extent)

            for range in config.range:
                min, max = range.min, range.max
                temp = np.where(np.logical_and(data >= min, data < max),
                                range.value, 0)

                if config.extent == ALL:
                    processed += temp
                    continue

                processed[mask] += temp[mask]

            return processed

        # No range, has to be map. Assign cells values based on map.
        temp = np.zeros(self._io_handler.shape, dtype=self._dtype)
        for key, val in config.map.items():  # type: ignore[union-attr]
            temp[data == key] = val

        if config.extent == ALL:
            return temp

        mask = self._get_mask(config.extent)
        processed = np.zeros(self._io_handler.shape, dtype=self._dtype)
        processed[mask] = temp[mask]
        return processed

    def _process_vector_layer(self, fname: str, config: FBLayerConfig
                              ) -> npt.NDArray:
        """
        Rasterize a vector layer

        Parameters
        ----------
        fname : str
            Name of vector layer to rasterize
        config : FBLayerConfig
            Config for layer

        Returns
        -------
        array-like
            Rasterized vector
        """
        if config.rasterize is None:
            raise ValueError(f'{fname} is a vector but the config is missing '
                             f'key "rasterize": {config}')

        r_config = config.rasterize

        temp = rasterize(fname, self._io_handler.profile,
                         buffer_dist=r_config.buffer,
                         burn_value=r_config.value,
                         dtype=self._dtype,
                         reproject_vector=r_config.reproject)

        if config.extent == ALL:
            return temp

        mask = self._get_mask(config.extent)
        processed = np.zeros(self._io_handler.shape, dtype=self._dtype)
        processed[mask] = temp[mask]
        return processed

    def _process_forced_inclusions(self, data: npt.NDArray,
                                   fi_layers: Dict[str, FBLayerConfig]
                                   ) -> npt.NDArray:
        """
        Use forced inclusion (FI) layers to remove barriers/friction. Any
        value > 0 in the FI layers will result in a 0 in the corresponding
        cell in the returned raster.

        Parameters
        ----------
        data : array-like
            Composite friction or barrier layer
        fi_layers : dict
            Dict of forced inclusions layers keyed by GeoTIFF filename.

        Returns
        -------
        array-like
            Composite layer with forced inclusions added
        """
        fi = np.zeros(self._io_handler.shape)

        for fname, config in fi_layers.items():
            if Path(fname).suffix.lower() not in ['.tif', '.tiff']:
                msg = (f'Forced inclusion file {fname} does not end with .tif.'
                       ' GeoTIFFs are the only format allowed for forced '
                       'inclusions.')
                logger.error(msg)
                raise ValueError(msg)

            map_given = config.map is not None
            range_given = config.range is not None
            rasterize_given = config.rasterize is not None
            if map_given or range_given or rasterize_given:
                msg = ('`map`, `range`, and `rasterize` are not allowed if '
                       '`forced_inclusion` is True, but one was found in '
                       f'config: {fname}: {config}')
                logger.error(msg)
                raise ValueError(msg)

            # Past guard clauses, process FI
            if config.extent != ALL:
                mask = self._get_mask(config.extent)

            temp = self._io_handler.load_tiff(fname, reproject=True)

            if config.extent == ALL:
                fi += temp
            else:
                fi[mask] += temp[mask]

        data[fi > 0] = 0
        return data

    def _get_mask(self, extent: Extents) -> MaskArr:
        """
        Get mask by requested extent

        Parameters
        ----------
        extent : Extents
            Extent of desired mask, 'all' is not allowed.

        Returns
        -------
        MaskArr
            Mask array
        """
        if extent == ALL:
            raise AttributeError(f'Mask for extent of {extent} is unnecessary')

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
    def _check_tiff_layer_config(config: FBLayerConfig):
        """
        Check if a FBLayerConfig is valid for a GeoTIFF.

        Parameters
        ----------
        config : FBLayerConfig
            The config model to check

        Raises
        ------
        ValueError
            If an issue is detected
        """
        if config.rasterize is not None:
            raise ValueError('"rasterize" is only for vectors. Found in '
                             f'config {config}')

        if config.map is not None and config.range is not None:
            raise ValueError('Keys "map" and "range" are mutually exclusive '
                             'but more than one was found in raster config '
                             f'{config}')

        if config.map is None and config.range is None:
            raise ValueError('Either "map" or "range" must be specified for '
                             'a raster, but neither were found in config '
                             f'{config}')
