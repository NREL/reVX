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
from reVX.handlers.layered_h5 import LayeredH5, check_geotiff
from reVX.handlers.outputs import Outputs

logger = logging.getLogger(__name__)


class DistToPortsConverter(LayeredH5):
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
            with Geotiff(geotiff, chunks=chunks) as tif:
                v = tif.values
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

    def write_dist_to_ports_to_h5(self, dist_to_ports, layer_name,
                                  check_tiff=True, transform_atol=0.01,
                                  description=None, scale_factor=None,
                                  dtype='float32'):
        """
        Transfer geotiff exclusions to h5 confirming they match existing layers

        Parameters
        ----------
        dist_to_ports : str
            Path to directory containing distance to port geotiff files or
            a list of the distance to port geotiff filepaths
        layer_name : str
            Layer to create
        check_tiff : bool, optional
            Flag to check tiff profile and coordinates against exclusion .h5
            profile and coordinates, by default True
        transform_atol : float, optional
            Absolute tolerance parameter when comparing geotiff transform data,
            by default 0.01
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

        if not os.path.exists(self.h5_file):
            if self.template_file == self.h5_file:
                self.template_file = dist_to_ports[0]
            self.create_new(overwrite=False)

        if check_tiff:
            check_geotiff(self, dist_to_ports[0], chunks=self._chunks,
                          transform_atol=transform_atol)

        with Geotiff(dist_to_ports[0], chunks=self._chunks) as tif:
            profile = tif.profile

        dist_to_ports = self._parse_dist_to_ports(dist_to_ports,
                                                  chunks=self._chunks)
        if layer_name in self.layers:
            msg = ("{} is already present in {} and will be updated"
                   .format(layer_name, self.h5_file))
            logger.warning(msg)
            warn(msg)
            with ExclusionLayers(self.h5_file) as exc:
                dist_to_ports = np.minimum(dist_to_ports, exc[layer_name])

        if scale_factor is not None:
            attrs = {'scale_factor': scale_factor}
            dist_to_ports = Outputs._check_data_dtype(
                layer_name, dist_to_ports, dtype, attrs=attrs)

        self.write_layer_to_h5(dist_to_ports, layer_name, profile=profile,
                               description=description,
                               scale_factor=scale_factor)

    def layers_to_h5(self, layers, check_tiff=True, transform_atol=0.01,
                     descriptions=None, scale_factors=None):
        """
        Create exclusions .h5 file, or load layers into existing exclusion .h5
        file from provided dist_to_ports

        Parameters
        ----------
        layers : dict
            Dictionary mapping goetiffs to the layers to load
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

        logger.info('Moving layers to %s', self.h5_file)
        for layer_name, dist_to_ports in layers.items():
            logger.info('- Transfering %s', layer_name)
            description = descriptions.get(layer_name, None)
            scale = scale_factors.get(layer_name, None)
            if scale is not None:
                scale_factor = scale['scale_factor']
                dtype = scale['dtype']
            else:
                scale_factor = None
                dtype = None

            self.write_dist_to_ports_to_h5(dist_to_ports, layer_name,
                                           check_tiff=check_tiff,
                                           transform_atol=transform_atol,
                                           description=description,
                                           scale_factor=scale_factor,
                                           dtype=dtype)
