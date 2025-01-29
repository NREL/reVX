"""
Build friction or barrier layers from raster and vector data.
"""
import logging
from pathlib import Path
from typing import Dict
from warnings import warn

import numpy as np
import numpy.typing as npt

from reVX.handlers.layered_h5 import LayeredTransmissionH5
from reVX.config.transmission_layer_creation import (Extents,
                                                     LayerBuildConfig,
                                                     LayerBuildComponents)
from reVX.least_cost_xmission.layers.base import BaseLayerCreator
from reVX.least_cost_xmission.layers.utils import rasterize_shape_file
from reVX.least_cost_xmission.layers.masks import MaskArr, Masks
from reVX.least_cost_xmission.config.constants import (DEFAULT_DTYPE, ALL,
                                                       METERS_IN_MILE,
                                                       CELL_SIZE)

logger = logging.getLogger(__name__)
TIFF_EXTENSIONS = {'.tif', '.tiff'}
SHP_EXTENSIONS = {'.shp', '.gpkg'}


class LayerCreator(BaseLayerCreator):
    """Build layer based on tiff and user config."""

    def __init__(self, io_handler: LayeredTransmissionH5,
                 masks: Masks, output_tiff_dir=".",
                 dtype: npt.DTypeLike=DEFAULT_DTYPE,
                 cell_size=CELL_SIZE):
        """
        Parameters
        ----------
        io_handler : :class:`LayeredTransmissionH5`
            Transmission layer IO handler
        masks : Masks
            Masks instance that can be used to retrieve multiple types
            of masks.
        output_tiff_dir : path-like, optional
            Directory where cost layers should be saved as GeoTIFF.
            By default, ``"."``.
        dtype : np.dtype, optional
            Data type for final dataset. By default, ``float32``.
        cell_size : int, optional
            Side length of each cell, in meters. Cells are assumed to be
            square. By default, :obj:`CELL_SIZE`.
        """
        self._masks = masks
        super().__init__(io_handler=io_handler, mask=None,
                         output_tiff_dir=output_tiff_dir, dtype=dtype,
                         cell_size=cell_size)

    def build(self, layer_name, build_config: LayerBuildComponents,
              values_are_costs_per_mile=False, write_to_h5=True,
              description=None):
        """
        Combine multiple GeoTIFFs and vectors to a raster layer and save
        to GeoTIFF/HDF5 file.

        Parameters
        ----------
        layer_name : str
            Name of layer to use in H5 and for output tiff.
        build_config : LayerBuildComponents
            Dict of LayerBuildConfig keyed by GeoTIFF/vector filenames.
        values_are_costs_per_mile : bool, default=False
            Option to convert values into costs per cell under the
            assumption that the resulting values arte costs in $/mile.
            By default, ``False``, which writes raw values to TIFF/H5.
        write_to_h5 : bool, default=True
            Option to write the layer to H5 file after creation.
            By default, ``True``.
        description : str, optional
            Optional description to store with this layer in the H5
            file. By default, ``None``.
        """
        layer_name = layer_name.replace(".tif", "").replace(".tiff", "")
        logger.debug('Combining %s layers', layer_name)
        result = np.zeros(self._io_handler.shape, dtype=self._dtype)
        fi_layers: LayerBuildComponents = {}

        for fname, config in build_config.items():
            if config.forced_inclusion:
                fi_layers[fname] = config
                continue

            logger.debug('Processing %s with config %s', fname, config)
            if Path(fname).suffix.lower() in TIFF_EXTENSIONS:
                data = self._io_handler.load_data_using_h5_profile(
                    fname, reproject=True)
                temp = self._process_raster_layer(data, config)
                result += temp
            elif Path(fname).suffix.lower() in SHP_EXTENSIONS:
                temp = self._process_vector_layer(fname, config)
                result += temp
            else:
                raise ValueError(f'Unsupported file extension on {fname}')

        result = self._process_forced_inclusions(result, fi_layers)
        if values_are_costs_per_mile:
            result = result / METERS_IN_MILE * self._cell_size

        out_filename = self.output_tiff_dir / f"{layer_name}.tif"
        logger.debug('Writing combined %s layers to %s', layer_name,
                     out_filename)
        self._io_handler.save_data_using_h5_profile(result, out_filename)
        if write_to_h5:
            out = self._io_handler.load_data_using_h5_profile(
                out_filename, reproject=True)
            logger.debug('Writing %s to H5', layer_name)
            self._io_handler.write_layer_to_h5(out, layer_name,
                                               description=description)

    def _process_raster_layer(self, data: npt.NDArray,  # type: ignore[return]
                              config: LayerBuildConfig) -> npt.NDArray:
        """Create the desired layer from the array using LayerBuildConfig.

        Desired "range" or "map" operation is only applied to the area
        indicated by the "extent".

        Parameters
        ----------
        data : array-like
            Array of data to process.
        config : LayerBuildConfig
            Definition of layer processing.

        Returns
        -------
        array-like | None
            Transformed data.
        """
        self._check_tiff_layer_config(config)

        if config.global_value is not None:
            temp = np.full(self._io_handler.shape,
                           fill_value=config.global_value,
                           dtype=self._dtype)

            if config.extent == ALL:
                return temp

            mask = self._get_mask(config.extent)
            processed = np.zeros(self._io_handler.shape, dtype=self._dtype)
            processed[mask] = temp[mask]
            return processed

        # Assign all cells one or more ranges to a value
        if config.bins is not None:
            _validate_bin_range(config.bins)
            _validate_bin_continuity(config.bins)

            processed = np.zeros(self._io_handler.shape, dtype=self._dtype)
            if config.extent != ALL:
                mask = self._get_mask(config.extent)

            for i, interval in enumerate(config.bins):
                logger.debug('Calculating layer values for bin '
                             f'{i+1}/{len(config.bins)}: {interval}')
                min, max = interval.min, interval.max
                temp = np.where(np.logical_and(data >= min, data < max),
                                interval.value, 0)

                if config.extent == ALL:
                    processed += temp
                    continue

                processed[mask] += temp[mask]

            return processed

        if config.pass_through:
            if config.extent == ALL:
                return data

            mask = self._get_mask(config.extent)
            processed = np.zeros(self._io_handler.shape, dtype=self._dtype)
            processed[mask] = data[mask]
            return processed

        # No bins specified, has to be map. Assign cells values based on map.
        temp = np.zeros(self._io_handler.shape, dtype=self._dtype)
        for key, val in config.map.items():  # type: ignore[union-attr]
            temp[data == key] = val

        if config.extent == ALL:
            return temp

        mask = self._get_mask(config.extent)
        processed = np.zeros(self._io_handler.shape, dtype=self._dtype)
        processed[mask] = temp[mask]
        return processed

    def _process_vector_layer(self, fname: str, config: LayerBuildConfig
                              ) -> npt.NDArray:
        """
        Rasterize a vector layer

        Parameters
        ----------
        fname : str
            Name of vector layer to rasterize
        config : LayerBuildConfig
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

        temp = rasterize_shape_file(fname, self._io_handler.profile,
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
                                   fi_layers: LayerBuildComponents
                                   ) -> npt.NDArray:
        """
        Use forced inclusion (FI) layers to remove barriers/friction. Any
        value > 0 in the FI layers will result in a 0 in the corresponding
        cell in the returned raster.

        Parameters
        ----------
        data : array-like
            Composite friction or barrier layer
        fi_layers : LayerBuildComponents
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

            global_value_given = config.global_value is not None
            map_given = config.map is not None
            range_given = config.bins is not None
            rasterize_given = config.rasterize is not None
            bad_input_given = (global_value_given
                               or map_given
                               or range_given
                               or rasterize_given)
            if bad_input_given:
                msg = ('`global_value`, `map`, `bins`, and `rasterize` are '
                       'not allowed if `forced_inclusion` is True, but one '
                       f'was found in config: {fname}: {config}')
                logger.error(msg)
                raise ValueError(msg)

            # Past guard clauses, process FI
            if config.extent != ALL:
                mask = self._get_mask(config.extent)

            temp = self._io_handler.load_data_using_h5_profile(fname,
                                                               reproject=True)

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
    def _check_tiff_layer_config(config: LayerBuildConfig):
        """ Check if a LayerBuildConfig is valid for a GeoTIFF.

        Parameters
        ----------
        config : LayerBuildConfig
            The config model to check

        Raises
        ------
        ValueError
            If an issue is detected
        """
        if config.rasterize is not None:
            raise ValueError('"rasterize" is only for vectors. Found in '
                             f'config {config}')

        mutex_entries = [config.map, config.bins, config.global_value]
        num_entries = sum(entry is not None for entry in mutex_entries)
        num_entries += int(config.pass_through)
        if num_entries > 1:
            raise ValueError('Keys "global_value", "map", "bins", and '
                             '"pass_through" are mutually exclusive but '
                             'more than one was found in raster config '
                             f'{config}')

        if num_entries < 1:
            raise ValueError('Either "global_value", "map", "bins", and '
                             '"pass_through" must be specified fora raster, '
                             'but none were found in config '
                             f'{config}')


def _validate_bin_range(bins):
    """Check for correctness in bin range. """
    for bin in bins:
        if bin.min > bin.max:
            raise AttributeError('Min is greater than max for bin config '
                                 f'{bin}.')
        if bin.min == float('-inf') and bin.max == float('inf'):
            msg = ('Bin covers all possible values, did you forget to set '
                   f'min or max? {bin}')
            logger.warning(msg)
            warn(msg)


def _validate_bin_continuity(bins):
    """Warn user of potential gaps in bin range continuity."""
    sorted_bins = sorted(bins, key=lambda x: x.min)
    last_max = float('-inf')
    for i, bin in enumerate(sorted_bins):
        if bin.min < last_max:
            last_bin = sorted_bins[i - 1] if i > 0 else '-infinity'
            msg = (f'Overlapping bins detected between bin {last_bin} '
                   f'and {bin}')
            logger.warning(msg)
            warn(msg)
        if bin.min > last_max:
            last_bin = sorted_bins[i - 1] if i > 0 else '-infinity'
            msg = f'Gap detected between bin {last_bin} and {bin}'
            logger.warning(msg)
            warn(msg)
        if i + 1 == len(sorted_bins):
            if bin.max < float('inf'):
                msg = f'Gap detected between bin {bin} and infinity'
                logger.warning(msg)
                warn(msg)

        last_max = bin.max
