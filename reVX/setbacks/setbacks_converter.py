# -*- coding: utf-8 -*-
"""
Convert setback geotiff to .h5 exclusion layers
"""
import logging
import numpy as np
import os
from warnings import warn

from reVX.handlers.geotiff import Geotiff
from reVX.handlers.layered_h5 import LayeredH5, check_geotiff
from reVX.handlers.outputs import Outputs

logger = logging.getLogger(__name__)


def parse_setbacks(setbacks, chunks=(128, 128), is_inclusion_layer=False):
    """Load setbacks, combine multiple setbacks by state if needed

    Parameters
    ----------
    setbacks : list
        List of paths to setback geotiffs to load and combine
    chunks : tuple, optional
        Chunk size of exclusions in Geotiff, by default (128, 128)
    is_inclusion_layer : bool, optional
        Flag indicating wether this layer should be an inclusion
        layer instead of an exclusion mask, by default False.

    Returns
    -------
    values : ndarray
        Setbacks exclusion array
    """
    logger.info("Merging setbacks...")
    logger.debug("\n\t- ".join([""] + setbacks))
    values = None
    reduction_func = np.minimum if is_inclusion_layer else np.maximum
    for geotiff in setbacks:
        with Geotiff(geotiff, chunks=chunks) as tif:
            data = tif.values

        if values is None:
            values = data
        else:
            values = reduction_func(values, data)

    return values


class SetbacksConverter(LayeredH5):
    """
    Convert setbacks goetiff(s) to excl .h5 layers
    """

    def write_setbacks_to_h5(self, setbacks, layer_name, check_tiff=True,
                             is_inclusion_layer=False, transform_atol=0.01,
                             description=None, scale_factor=None,
                             dtype='uint8', replace=True):
        """
        Transfer geotiff setbacks to h5 confirming they match existing layers

        Parameters
        ----------
        setbacks : str
            Path to geotiff file or directory containing multiple
            geotiff files to be merged.
        layer_name : str
            Name of layer to create of replace.
        check_tiff : bool, optional
            Flag to check tiff profile and coordinates against exclusion
            .h5 profile and coordinates, by default True.
        is_inclusion_layer : bool, optional
            Flag indicating wether this layer should be an inclusion
            layer instead of an exclusion mask, by default False.
        transform_atol : float, optional
            Absolute tolerance parameter when comparing geotiff
            transform data, by default 0.01
        coord_atol : float, optional.
            Absolute tolerance parameter when comparing new un-projected
            geotiff coordinates against previous coordinates,
            by default 0.001.
        description : str, optional
            Description of exclusion layer, by default None.
        scale_factor : int | float, optional
            Scale factor to use to scale geotiff data when added to the
            .h5 file, by default None, which does not apply any scaling.
        dtype : str, optional
            Dtype to save geotiff data as in the .h5 file. Only used
            when 'scale_factor' is not None, by default 'uint8',
        replace : bool, optional
            Option to replace existing layer (if any).
            By default, ``True``.
        """
        if os.path.isdir(setbacks):
            setbacks = [os.path.join(setbacks, file)
                        for file in os.listdir(setbacks)
                        if file.endswith('.tif')]
        else:
            setbacks = [setbacks]

        logger.debug('\t- Combining setbacks in {}'.format(setbacks))

        if not os.path.exists(self.h5_file):
            if self.template_file == self.h5_file:
                self.template_file = setbacks[0]
            self.create_new(overwrite=False)

        self._warn_or_error_for_existing_layer(layer_name, replace)

        if check_tiff:
            check_geotiff(self, setbacks[0], chunks=self._chunks,
                          transform_atol=transform_atol)

        with Geotiff(setbacks[0], chunks=self._chunks) as tif:
            profile = tif.profile

        setbacks = parse_setbacks(setbacks, chunks=self._chunks,
                                  is_inclusion_layer=is_inclusion_layer)
        if scale_factor is not None:
            attrs = {'scale_factor': scale_factor}
            setbacks = Outputs._check_data_dtype(layer_name, setbacks, dtype,
                                                 attrs=attrs)

        logger.debug('Writing final setback layer to %s', self.h5_file)
        self.write_layer_to_h5(setbacks, layer_name, profile=profile,
                               description=description,
                               scale_factor=scale_factor)

    def layers_to_h5(self, layers, replace=True, check_tiff=True,
                     are_inclusion_layers=False, transform_atol=0.01,
                     descriptions=None, scale_factors=None):
        """
        Create exclusions .h5 file, or load layers into existing exclusion .h5
        file from provided setbacks

        Parameters
        ----------
        layers : dict | list
            Dictionary where keys are layer names and values are paths
            to the corresponding geotiff files or paths to directories
            containing multiple geotiff files to be merged for each
            layer. If input is a list of paths to geotiff files, then
            the name of the layer is inferred from the geotiff file
            name.
        replace : bool, optional
            Flag to replace existing layers if needed, by default True
        check_tiff : bool, optional
            Flag to check tiff profile and coordinates against exclusion
            .h5 profile and coordinates, by default True.
        are_inclusion_layers : bool, optional
            Flag indicating wether the input layers should be treated
            as inclusion layers instead of exclusion masks,
            by default False.
        transform_atol : float, optional
            Absolute tolerance parameter when comparing geotiff
            transform data, by default 0.01.
        description : dict, optional
            Description of exclusion layers, by default None
        scale_factor : dict, optional
            Scale factors and dtypes to use when scaling given layers,
            by default None
        """
        if isinstance(layers, list):
            layers = {os.path.basename(lyr).split('.')[0]: lyr
                      for lyr in layers}

        if scale_factors is None:
            scale_factors = {}

        if descriptions is None:
            descriptions = {}

        logger.info('Moving layers to %s', self.h5_file)
        for layer_name, setbacks in layers.items():
            logger.info('- Transfering %s', layer_name)
            description = descriptions.get(layer_name, None)
            scale = scale_factors.get(layer_name, None)
            if scale is not None:
                scale_factor = scale['scale_factor']
                dtype = scale['dtype']
            else:
                scale_factor = None
                dtype = None

            self.write_setbacks_to_h5(setbacks, layer_name,
                                      check_tiff=check_tiff,
                                      is_inclusion_layer=are_inclusion_layers,
                                      transform_atol=transform_atol,
                                      description=description,
                                      scale_factor=scale_factor,
                                      dtype=dtype, replace=replace)
