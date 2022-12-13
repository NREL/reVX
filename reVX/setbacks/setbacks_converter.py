# -*- coding: utf-8 -*-
"""
Convert setback geotiff to .h5 exclusion layers
"""
import logging
import numpy as np
import os
from warnings import warn

from reVX.handlers.geotiff import Geotiff
from reVX.handlers.outputs import Outputs
from reVX.utilities import ExclusionsConverter

logger = logging.getLogger(__name__)


def parse_setbacks(setbacks, chunks=(128, 128), is_inclusion_layer=False):
    """Load setbacks, combine multiple setbacks by state if needed

    This method assumes the data in separate files is non-overlapping.
    In other words, a file containing setbacks exclusions for Illinois
    should not contain any exclusions for Indiana, assuming the setbacks
    for Indiana are in a separate tif file in the same directory.

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
        data = ExclusionsConverter.parse_tiff(geotiff, chunks=chunks,
                                              check_tiff=False)[1]
        if values is None:
            values = data
        else:
            values = reduction_func(values, data)

    return values


class SetbacksConverter(ExclusionsConverter):
    """
    Convert setbacks goetiff(s) to excl .h5 layers
    """

    def setbacks_to_layer(self, layer, setbacks, check_tiff=True,
                          is_inclusion_layer=False, transform_atol=0.01,
                          coord_atol=0.001, description=None,
                          scale_factor=None, dtype='uint8'):
        """
        Transfer geotiff setbacks to h5 confirming they match existing layers

        Parameters
        ----------
        layer : str
            Name of layer to create of replace.
        setbacks : str
            Path to geotiff file or directory containing multiple
            geotiff files to be merged.
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
            when 'scale_factor' is not None, by default 'uint8'
        """
        if os.path.isdir(setbacks):
            setbacks = [os.path.join(setbacks, file)
                        for file in os.listdir(setbacks)
                        if file.endswith('.tif')]
        else:
            setbacks = [setbacks]

        logger.debug('\t- Combining setbacks in {}'.format(setbacks))

        if not os.path.exists(self._excl_h5):
            self._init_h5(self._excl_h5, setbacks[0], chunks=self._chunks)

        msg = ("{} is already present in {}".format(layer, self._excl_h5))
        if layer in self.layers:
            if self._replace:
                msg += " and will be replaced"
                logger.warning(msg)
                warn(msg)
            else:
                msg += ", to 'replace' set to True"
                logger.error(msg)
                raise KeyError(msg)

        if check_tiff:
            self._check_geotiff(self._excl_h5, setbacks[0],
                                chunks=self._chunks,
                                transform_atol=transform_atol,
                                coord_atol=coord_atol)

        with Geotiff(setbacks[0], chunks=self._chunks) as tif:
            profile = tif.profile

        setbacks = parse_setbacks(setbacks, chunks=self._chunks,
                                  is_inclusion_layer=is_inclusion_layer)
        if scale_factor is not None:
            setbacks = Outputs._check_data_dtype(setbacks, dtype,
                                                 scale_factor=scale_factor)

        logger.debug('Writing final setback layer to {!r}'
                     .format(self._excl_h5))
        self._write_layer(self._excl_h5, layer, profile, setbacks,
                          chunks=self._chunks, description=description,
                          scale_factor=scale_factor)

    @classmethod
    def layers_to_h5(cls, excl_h5, layers, chunks=(128, 128),
                     replace=True, check_tiff=True,
                     are_inclusion_layers=False, transform_atol=0.01,
                     coord_atol=0.001, descriptions=None, scale_factors=None):
        """
        Create exclusions .h5 file, or load layers into existing exclusion .h5
        file from provided setbacks

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing or to contain exclusion layers
        layers : dict | list
            Dictionary where keys are layer names and values are paths
            to the corresponding geotiff files or paths to directories
            containing multiple geotiff files to be merged for each
            layer. If input is a list of paths to geotiff files, then
            the name of the layer is inferred from the geotiff file
            name.
        chunks : tuple, optional
            Chunk size of exclusions in geotiff, by default (128, 128)
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
        coord_atol : float, optional
            Absolute tolerance parameter when comparing new un-projected
            geotiff coordinates against previous coordinates,
            by default 0.001.
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

        excls = cls(excl_h5, chunks=chunks, replace=replace)
        logger.info('Creating {}'.format(excl_h5))
        for layer, setbacks in layers.items():
            logger.info('- Transfering {}'.format(layer))
            scale = scale_factors.get(layer, None)
            if scale is not None:
                scale_factor = scale['scale_factor']
                dtype = scale['dtype']
            else:
                scale_factor = None
                dtype = None

            description = descriptions.get(layer, None)

            excls.setbacks_to_layer(layer, setbacks, check_tiff=check_tiff,
                                    is_inclusion_layer=are_inclusion_layers,
                                    transform_atol=transform_atol,
                                    coord_atol=coord_atol,
                                    description=description,
                                    scale_factor=scale_factor,
                                    dtype=dtype)
