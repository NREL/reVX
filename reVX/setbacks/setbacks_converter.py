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
from reVX.utilities.exclusions_converter import ExclusionsConverter

logger = logging.getLogger(__name__)


class SetbacksConverter(ExclusionsConverter):
    """
    Convert setbacks goetiff(s) to excl .h5 layers
    """
    @classmethod
    def _parse_setbacks(cls, setbacks, chunks=(128, 128)):
        """
        Load setbacks, combine multiple setbacks by state if needed

        Parameters
        ----------
        setbacks : list
            List of paths to setback geotiffs to load and combine
        chunks : tuple, optional
            Chunk size of exclusions in Geotiff, by default (128, 128)

        Returns
        -------
        values : ndarray
            Setbacks exclusion array
        """
        values = None
        for geotiff in setbacks:
            v = cls._parse_tiff(geotiff, chunks=chunks, check_tiff=False)[1]
            if values is None:
                values = v
            else:
                values = np.maximum(values, v)

        return values

    def setbacks_to_layer(self, layer, setbacks, check_tiff=True,
                          transform_atol=0.01, coord_atol=0.001,
                          description=None, scale_factor=None, dtype='uint8'):
        """
        Transfer geotiff exclusions to h5 confirming they match existing layers

        Parameters
        ----------
        layer : str
            Layer to extract
        geotiff : str
            Path to geotiff file
        check_tiff : bool, optional
            Flag to check tiff profile and coordinates against exclusion .h5
            profile and coordinates, by default True
        transform_atol : float, optional
            Absolute tolerance parameter when comparing geotiff transform data,
            by default 0.01
        coord_atol : float, optional
            Absolute tolerance parameter when comparing new un-projected
            geotiff coordinates against previous coordinates, by default 0.001
        description : str, optional
            Description of exclusion layer, by default None
        scale_factor : int | float, optional
            Scale factor to use to scale geotiff data when added to the .h5
            file, by default None
        dtype : str, optional
            Dtype to save geotiff data as in the .h5 file. Only used when
            'scale_factor' is not None, by default 'uint8'
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

        setbacks = self._parse_setbacks(setbacks, chunks=self._chunks)
        if scale_factor is not None:
            setbacks = Outputs._check_data_dtype(setbacks, dtype,
                                                 scale_factor=scale_factor)

        self._write_layer(self._excl_h5, layer, profile, setbacks,
                          chunks=self._chunks, description=description,
                          scale_factor=scale_factor)

    @classmethod
    def layers_to_h5(cls, excl_h5, layers, chunks=(128, 128),
                     replace=True, check_tiff=True,
                     transform_atol=0.01, coord_atol=0.001,
                     descriptions=None, scale_factors=None):
        """
        Create exclusions .h5 file, or load layers into existing exclusion .h5
        file from provided setbacks

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing or to contain exclusion layers
        layers : dict
            Dictionary mapping goetiffs to the layers to load
        chunks : tuple, optional
            Chunk size of exclusions in Geotiff, by default (128, 128)
        replace : bool, optional
            Flag to replace existing layers if needed, by default True
        check_tiff : bool, optional
            Flag to check tiff profile and coordinates against exclusion .h5
            profile and coordinates, by default True
        transform_atol : float, optional
            Absolute tolerance parameter when comparing geotiff transform data,
            by default 0.01
        coord_atol : float, optional
            Absolute tolerance parameter when comparing new un-projected
            geotiff coordinates against previous coordinates, by default 0.001
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
                                    transform_atol=transform_atol,
                                    coord_atol=coord_atol,
                                    description=description,
                                    scale_factor=scale_factor,
                                    dtype=dtype)
