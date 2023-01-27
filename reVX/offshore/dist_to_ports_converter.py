# -*- coding: utf-8 -*-
"""
Convert dist_to_ports geotiff to .h5 exclusion layers
"""
import logging
import numpy as np
import os
from warnings import warn

from reV.handlers.exclusions import ExclusionLayers

from reVX.handlers.geotiff import Geotiff
from reVX.handlers.outputs import Outputs
from reVX.utilities import ExclusionsConverter

logger = logging.getLogger(__name__)


class DistToPortsConverter(ExclusionsConverter):
    """
    Convert Distance to Ports goetiff(s) to excl .h5 layers
    """
    @classmethod
    def _parse_dist_to_ports(cls, dist_to_ports, chunks=(128, 128)):
        """
        Load dist_to_ports, combine multiple dist_to_ports by state if needed

        Parameters
        ----------
        dist_to_ports : list
            List of paths to dist_to_ports geotiffs to load and combine
        chunks : tuple, optional
            Chunk size of exclusions in Geotiff, by default (128, 128)

        Returns
        -------
        values : ndarray
            dist_to_ports exclusion array
        """
        values = None
        for geotiff in dist_to_ports:
            v = cls.parse_tiff(geotiff, chunks=chunks, check_tiff=False)[1]
            if not np.any(v):
                msg = ('{} is invalid and only contains zeros and will be '
                       'skipped. It is advised to recreate the file and '
                       'update the distance to ports layer!'
                       .format(geotiff))
                logger.warning(msg)
                warn(msg)
            else:
                if values is None:
                    values = v
                else:
                    values = np.minimum(values, v)

        return values

    def dist_to_ports_to_layer(self, layer, dist_to_ports, check_tiff=True,
                               transform_atol=0.01, coord_atol=0.001,
                               description=None, scale_factor=None,
                               dtype='float32'):
        """
        Transfer geotiff exclusions to h5 confirming they match existing layers

        Parameters
        ----------
        layer : str
            Layer to create
        dist_to_ports : str
            Path to directory containing distance to port geotiff files or
            a list of the distance to port geotiff filepaths
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
            'scale_factor' is not None, by default 'float32'
        """
        if os.path.isdir(dist_to_ports):
            dist_to_ports = [os.path.join(dist_to_ports, file)
                             for file in os.listdir(dist_to_ports)
                             if file.endswith('.tif')]
        else:
            dist_to_ports = [dist_to_ports]

        logger.debug('\t- Computing minimum distance to ports from {}'
                     .format(dist_to_ports))

        if not os.path.exists(self._excl_h5):
            self._init_h5(self._excl_h5, dist_to_ports[0],
                          chunks=self._chunks)

        if check_tiff:
            self._check_geotiff(self._excl_h5, dist_to_ports[0],
                                chunks=self._chunks,
                                transform_atol=transform_atol,
                                coord_atol=coord_atol)

        with Geotiff(dist_to_ports[0], chunks=self._chunks) as tif:
            profile = tif.profile

        dist_to_ports = self._parse_dist_to_ports(dist_to_ports,
                                                  chunks=self._chunks)
        if layer in self.layers:
            msg = ("{} is already present in {} and will be updated"
                   .format(layer, self._excl_h5))
            logger.warning(msg)
            warn(msg)
            with ExclusionLayers(self._excl_h5) as exc:
                dist_to_ports = np.minimum(dist_to_ports, exc[layer])

        if scale_factor is not None:
            dist_to_ports = Outputs._check_data_dtype(
                dist_to_ports, dtype, scale_factor=scale_factor)

        self._write_layer(self._excl_h5, layer, profile, dist_to_ports,
                          chunks=self._chunks, description=description,
                          scale_factor=scale_factor)

    @classmethod
    def layers_to_h5(cls, excl_h5, layers, chunks=(128, 128),
                     replace=True, check_tiff=True,
                     transform_atol=0.01, coord_atol=0.001,
                     descriptions=None, scale_factors=None):
        """
        Create exclusions .h5 file, or load layers into existing exclusion .h5
        file from provided dist_to_ports

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
        if scale_factors is None:
            scale_factors = {}

        if descriptions is None:
            descriptions = {}

        if isinstance(layers, list):
            layers = {os.path.basename(lyr).split('.')[0]: lyr
                      for lyr in layers}

        excls = cls(excl_h5, chunks=chunks, replace=replace)
        logger.info('Creating {}'.format(excl_h5))
        for layer, dist_to_ports in layers.items():
            logger.info('- Transfering {}'.format(layer))
            scale = scale_factors.get(layer, None)
            if scale is not None:
                scale_factor = scale['scale_factor']
                dtype = scale['dtype']
            else:
                scale_factor = None
                dtype = None

            description = descriptions.get(layer, None)

            excls.dist_to_ports_to_layer(layer, dist_to_ports,
                                         check_tiff=check_tiff,
                                         transform_atol=transform_atol,
                                         coord_atol=coord_atol,
                                         description=description,
                                         scale_factor=scale_factor,
                                         dtype=dtype)
