# -*- coding: utf-8 -*-
"""
Abstract base calss for layer builders
"""
from pathlib import Path
from abc import ABC, abstractmethod

import numpy.typing as npt

from reVX.handlers.layered_h5 import LayeredTransmissionH5
from reVX.least_cost_xmission.config.constants import DEFAULT_DTYPE, CELL_SIZE


class BaseLayerCreator(ABC):
    """
    Abstract Base Class to create and save transmission routing layers
    """
    def __init__(self, io_handler: LayeredTransmissionH5,
                 mask=None, output_tiff_dir=".",
                 dtype: npt.DTypeLike = DEFAULT_DTYPE,
                 cell_size=CELL_SIZE):
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
        cell_size : int, optional
            Side length of each cell, in meters. Cells are assumed to be
            square. By default, :obj:`CELL_SIZE`.
        """
        self._io_handler = io_handler
        self.output_tiff_dir = Path(output_tiff_dir)
        self._mask = mask
        self._dtype = dtype
        self._cell_size = cell_size

    @property
    def shape(self):
        """tuple: Layer shape. """
        return self._io_handler.shape

    @abstractmethod
    def build(self, *args, **kwargs):
        """Build layer"""
        raise NotImplementedError
