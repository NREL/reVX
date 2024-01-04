import logging
from functools import reduce
from typing import Optional, Union, TypedDict, List

import numpy as np
import numpy.typing as npt
import rasterio as rio

from reVX.least_cost_xmission.offshore_utilities import CombineRasters, _sum

logger = logging.getLogger(__name__)

"""
Config for assigning cost based on bins. Cells with values >= than 'min' and <
'max' will be assigned 'cost'. One or both of 'min' and 'max' can be specified.
'cost' must be specified.
"""
BinConfig = TypedDict('BinConfig', {
    'min': float,
    'max': float,
    'cost': float,  # mandatory
},
    total=False
)

Mask = npt.NDArray[np.bool_]
OptionalMask = Optional[Mask]

class OffshoreCostCreator(CombineRasters):
    """
    Create offshore and combined offshore-land costs and barrier layers for
    least cost path transmission routing. This is implemented as a child class
    of CombineRasters to preserve that class's original functionality until it
    is deprecated.

    Note that the designation of offshore vs land is more refined in this
    implementation and landfall cells are handled separately from land.
    """
    OFFSHORE_COSTS_FNAME = 'offshore_costs.tif'

    LANDFALL_MASK_FNAME = 'landfall_mask.tif'  # One pixel width line at shore
    RAW_LAND_MASK_FNAME = 'raw_land_mask.tif'  # Rasterized land vector
    LAND_MASK_FNAME = 'land_mask.tif'  # = Raw mask - landfall mask
    OFFSHORE_MASK_FNAME = 'offshore_mask.tif'

    def __init__(self, template_f: str, layer_dir: str = '',
                 slope_barrier_cutoff: int = 15,):
        """
        Parameters
        ----------
        template_f
            Path to template raster with CRS to use for geopackage
        layer_dir
            Directory to prepend to barrier and cost layer filenames
        slope_barrier_cutoff
            Slopes >= this value are used as barriers.
        """
        super().__init__(template_f, layer_dir=layer_dir,
                         slope_barrier_cutoff=slope_barrier_cutoff)

        # offshore costs
        self._os_costs: Optional[npt.NDArray[np.float32]] = None

        # landfall mask, just the shore
        self._landfall_mask: OptionalMask = None
        # land mask raster, true indicates land
        self._land_mask: OptionalMask = None
        # offshore mask, true == water
        self._offshore_mask: OptionalMask = None

    def assign_cost_by_bins(self, in_filename: str, bins: List[BinConfig],
                                out_filename: str):
        """
        Assign costs based on binned raster values. Cells with values >= than
        'min' and < 'max' will be assigned 'cost'. One or both of 'min' and
        'max' can be specified. 'cost' must be specified.

        Parameters
        ----------
        in_filename
            Input raster to assign costs based upon.
        bins
            List of bins to use for assigning costs.
        out_filename
            Output raster with binned costs.
        """
        with rio.open(in_filename) as ras:
            input = ras.read(1)

        output = self._assign_values_by_bins(input, bins)
        self._save_tiff(output, out_filename)

    @staticmethod
    def _assign_values_by_bins(input: np.ndarray, bins: List[BinConfig]
                               ) -> np.ndarray:
        """
        Assign values based on binned raster values. Cells with values >= than
        'min' and < 'max' will be assigned 'cost'. One or both of 'min' and
        'max' can be specified. 'cost' must be specified.

        Parameters
        ----------
        input
            Input raster to assign values based upon.
        bins
            List of bins to use for assigning costs.

        Returns
        -------
            Binned costs
        """
        for bin in bins:
            if 'cost' not in bin:
                raise AttributeError(f'Bin config {bin} is missing "cost".')
            if ('min' not in bin) and ('max' not in bin):
                raise AttributeError(f'Bin config {bin} requires "min", "max",'
                                     ' or both.')
            if ('min' in bin) and ('max' in bin) and (bin['min'] > bin['max']):
                raise AttributeError('Min is greater than max for bin config '
                                     f'{bin}.')

        # Warn user of potential oversights in bin config. Look for gaps
        # between bin mins and maxes and overlapping bins.
        sorted_bins = sorted(bins, key=lambda x: x.get('min', float('-inf')))
        last_max = float('-inf')
        for i, bin in enumerate(sorted_bins):
            if bin.get('min', float('-inf')) < last_max:
                last_bin = sorted_bins[i - 1] if i > 0 else '-infinity'
                msg = (f'Overlapping bins detected between bin {last_bin} '
                       f'and {bin}')
                logger.warning(msg)
            if bin.get('min', float('-inf')) > last_max:
                last_bin = sorted_bins[i - 1] if i > 0 else '-infinity'
                msg = f'Gap detected between bin {last_bin} and {bin}'
                logger.warning(msg)

            if i + 1 == len(sorted_bins):
                if bin.get('max', float('inf')) < float('inf'):
                    msg = f'Gap detected between bin {bin} and infinity'
                    logger.warning(msg)

            last_max = bin.get('max', float('inf'))

        # Past guard clauses, perform binning
        output = np.zeros(input.shape)

        for i, bin in enumerate(bins):
            logger.debug(f'Calculating costs for bin {i+1}/{len(bins)}: {bin}')
            if ('min' in bin) and ('max' not in bin):
                output = np.where(input >= bin['min'], bin['cost'], output)
            elif ('min' not in bin) and ('max' in bin):
                output = np.where(input < bin['max'], bin['cost'], output)
            elif ('min' in bin) and ('max' in bin):
                mask = np.logical_and(input >= bin['min'], input < bin['max'])
                output = np.where(mask, bin['cost'], output)

        return output

    def create_masks(self, land_mask_shp_f: str, save_tiff: bool = False,
                     reproject_vector: bool = True):
        """
        Create the offshore and land mask layers from a polygon land vector
        file.

        Parameters
        ----------
        mask_shp_f
            Full path to land polygon gpgk or shp file
        save_tiff
            Save mask as tiff if true
        reproject_vector
            Reproject CRS of vector to match template raster if True.
        """
        logger.debug('Creating masks from %s', land_mask_shp_f)

        # Raw land is all land cells, include landfall cells
        raw_land = self.rasterize(land_mask_shp_f, all_touched=True,
                                  reproject_vector=reproject_vector)

        raw_land_mask: Mask = raw_land == 1

        # Offshore mask is inversion of raw land mask
        self._offshore_mask = ~raw_land_mask

        landfall = self.rasterize(land_mask_shp_f, all_touched=True,
                                  reproject_vector=reproject_vector,
                                  boundary_only=True)
        self._landfall_mask = landfall == 1

        # XOR landfall and raw land to get all land cells, except landfall
        # cells
        self._land_mask = np.logical_xor(self._landfall_mask, # type: ignore
                                         raw_land_mask)

        if save_tiff:
            logger.debug('Saving land and offshore masks to GeoTIFF')
            self._save_tiff(raw_land_mask, self.RAW_LAND_MASK_FNAME)
            self._save_tiff(self._offshore_mask, self.OFFSHORE_MASK_FNAME)
            self._save_tiff(self._land_mask, self.LAND_MASK_FNAME)
            self._save_tiff(self._landfall_mask, self.LANDFALL_MASK_FNAME)

    def load_masks(self):
        """
        Load the mask layers from GeoTIFFs. This does not need to be called if
        self.create_masks() was run previously. Mask files must be in the
        current directory.
        """
        def __load_mask(fname: str) -> npt.NDArray[np.bool_]:
            with rio.open(fname) as ras:
                raster = ras.read(1)

            assert raster.max() == 1
            assert raster.min() == 0
            assert raster.shape == self._os_shape
            return raster == 1

        self._land_mask = __load_mask(self.LAND_MASK_FNAME)
        self._offshore_mask = __load_mask(self.OFFSHORE_MASK_FNAME)
        self._landfall_mask = __load_mask(self.LANDFALL_MASK_FNAME)

        logger.info('Successfully loaded offshore and land masks')

    def build_off_shore_costs(self, cost_files: List[str],
                                save_tiff: bool = True,
                                dtype: npt.DTypeLike = 'float32'):
        """
        Additively combine off shore costs.

        Parameters
        ----------
        cost_files
            List of raster files to combine. All must have the same CRS and
            transform.
        save_tiff, optional
            Save combined cost raster to tiff if True, by default True
        dtype
            Numpy data type for combined data
        """
        layers: List[np.ndarray] = []

        for file in cost_files:
            data: np.ndarray = rio.open(file).read(1)
            assert data.shape == self._os_shape
            if data.min() < 0:
                raise ValueError(f'Cost layer {file} has values less than 0')
            layers.append(data)

        self._os_costs = reduce(_sum, layers).astype(dtype)

        if save_tiff:
            logger.info('Saving combined offshore costs to tiff')
            self._save_tiff(self._os_costs, self.OFFSHORE_COSTS_FNAME)


    # def load_land_masks(self, raw_land_mask: )