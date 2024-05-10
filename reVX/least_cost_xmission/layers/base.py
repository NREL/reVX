# -*- coding: utf-8 -*-
"""
Abstract base calss for layer builders
"""
from pathlib import Path
from abc import ABC, abstractmethod

import numpy.typing as npt

from reVX.handlers.layered_h5 import LayeredTransmissionH5
from reVX.least_cost_xmission.config.constants import DEFAULT_DTYPE


class BaseLayerCreator(ABC):
    """
    Abstract Base Class to create and save transmission routing layers
    """
    def __init__(self, io_handler: LayeredTransmissionH5,
                 mask=None, output_tiff_dir=".",
                 dtype: npt.DTypeLike = DEFAULT_DTYPE):
        """
        Parameters
        ----------
        io_handler : :class:`LayeredTransmissionH5`
            Transmission layer IO handler
        mask : ndarray, optional
            Array representing mask for layer values. Only optional if
            subclass implementation handles masks differently
            (e.g. the `LayerCreator` class). By default, ``None``
        output_tiff_dir : path-like, optional
            Directory where cost layers should be saved as GeoTIFF.
            By default, ``"."``.
        dtype : np.dtype, optional
            Data type for final dataset. By default, ``float32``.
        """
        self._io_handler = io_handler
        self.output_tiff_dir = Path(output_tiff_dir)
        self._mask = mask
        self._dtype = dtype

    @property
    def shape(self):
        """tuple: Layer shape. """
        return self._io_handler.shape

    @abstractmethod
    def build(self, *args, **kwargs):
        """Build layer"""
        raise NotImplementedError
