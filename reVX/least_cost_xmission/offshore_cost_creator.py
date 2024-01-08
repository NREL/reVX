"""
Create offshore costs. Merge offshore costs and barriers with land layers and
save to H5.
"""
import json
import logging
from functools import reduce
from typing import Optional, TypedDict, List

import h5py
import numpy as np
import numpy.typing as npt
import rasterio as rio
from rasterio.warp import reproject, Resampling

import rex
from reVX.least_cost_xmission.offshore_utilities import CombineRasters, _sum

logger = logging.getLogger(__name__)

H5_CHUNKS = (1, 128, 128)

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
    COMBO_COSTS_FNAME = 'combo_costs.tif'
    COMBO_BARRIERS_FNAME = 'combo_barriers.tif'

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

        self._offshore_costs: Optional[npt.NDArray[np.float32]] = None
        self._offshore_barriers: Optional[npt.NDArray[np.float32]] = None

        # landfall mask, just the shore
        self._landfall_mask: Optional[Mask] = None
        # land mask raster, true indicates land
        self._land_mask: Optional[Mask] = None
        # offshore mask, true == water
        self._offshore_mask: Optional[Mask] = None

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
    def _assign_values_by_bins(input: np.ndarray,  # noqa: C901
                               bins: List[BinConfig]) -> np.ndarray:
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
        self._land_mask = np.logical_xor(self._landfall_mask,  # type: ignore
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

    def load_offshore_costs(self, costs_file: str):
        """
        Load offshore costs from GeoTiff

        Parameters
        ----------
        costs_file
            Offshore costs file
        """

        with rio.open(costs_file) as ras:
            data: npt.NDArray = ras.read(1)
        assert data.shape == self._os_shape
        if data.min() < 0:
            raise ValueError(f'Costs layer {costs_file} has values less than '
                             '0')
        self._offshore_costs = data

    def load_offshore_barriers(self, barriers_file: Optional[str] = None):
        """
        Load offshore barriers from GeoTiff

        Parameters
        ----------
        barriers_file, optional
            Offshore costs file
        """
        if barriers_file is None:
            barriers_file = self.OFFSHORE_BARRIERS_FNAME

        with rio.open(barriers_file) as ras:
            data: npt.NDArray = ras.read(1)
        assert data.shape == self._os_shape
        if data.min() < 0:
            raise ValueError(
                f'Barriers layer {barriers_file} has values less than 0'
            )
        self._offshore_barriers = data

    def build_offshore_costs(self, cost_files: List[str],
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

        self._offshore_costs = reduce(_sum, layers).astype(dtype)

        if save_tiff:
            logger.info('Saving combined offshore costs to tiff')
            self._save_tiff(self._offshore_costs, self.OFFSHORE_COSTS_FNAME)

    def merge_os_and_land_costs(self, land_h5: str, h5_layer_name: str,
                                offshore_h5: str, landfall_cost: float,
                                save_tiff=False,):
        """
        Merge offshore and land costs and save to h5.

        Parameters
        ----------
        land_h5
            H5 file with land costs
        h5_layer_name
            Name of land costs layer in H5 file
        offshore_h5
            H5 file to write combined costs to
        landfall_cost
            Cost for landfall substation
        save_tiff, optional
            Save combined costs to GeoTIFF if True, by default False
        """
        # Sanity check that required data exists
        required_attrs = [
            '_offshore_costs', '_offshore_mask', '_land_mask', '_landfall_mask'
        ]
        for attr in required_attrs:
            if getattr(self, attr) is None:
                raise ValueError(f'Attribute self.{attr} has a value of None. '
                                 'Please run the appropriate method to '
                                 'populate it. ')
        # type narrowing
        assert self._offshore_costs is not None
        assert self._offshore_mask is not None
        assert self._landfall_mask is not None

        # Load land layer
        logger.info('Loading land costs "%s" from %s', h5_layer_name,
                    land_h5)
        with rex.Resource(land_h5) as res:
            land_profile_json = res.attrs[h5_layer_name]['profile']
            old_land_data = res[h5_layer_name][0]
        old_land_profile = json.loads(land_profile_json)

        # Reproject land barriers to new offshore projection
        logger.info('Reprojecting land costs')
        combo_costs = self._reproject(old_land_data, old_land_profile,
                                      dtype='float32')
        assert self._offshore_costs.shape == combo_costs.shape

        # Include offshore costs
        mask = self._offshore_mask
        combo_costs[mask] = self._offshore_costs[mask]

        # Landfall costs
        combo_costs[self._landfall_mask] = landfall_cost

        if save_tiff:
            logger.info('Saving offshore costs combined with %s to %s',
                        h5_layer_name, self.COMBO_COSTS_FNAME)
            self._save_tiff(combo_costs, self.COMBO_COSTS_FNAME)

        # Write to h5
        logger.info('Writing offshore costs combined with land "%s" to '
                    '%s', h5_layer_name, offshore_h5)
        combo_costs = combo_costs[np.newaxis, ...]

        self._write_to_h5(offshore_h5, h5_layer_name, combo_costs)

    def merge_os_and_land_barriers(self, land_h5: str, h5_layer_name: str,
                                   offshore_h5: str, save_tiff=False):
        """
        Merge offshore and land barriers and save to h5.

        Parameters
        ----------
        land_h5
            H5 file with land barriers
        h5_layer_name
            Name of land barriers layer in H5 file
        offshore_h5
            H5 file to write combined barriers to
        save_tiff, optional
            Save combined barriers to GeoTIFF if True, by default False
        """
        # Sanity check that required data exists
        required_attrs = [
            '_offshore_barriers',
            '_offshore_mask',
            '_land_mask',
            '_landfall_mask'
        ]
        for attr in required_attrs:
            if getattr(self, attr) is None:
                raise ValueError(f'Attribute self.{attr} has a value of None. '
                                 'Please run the appropriate method to '
                                 'populate it. ')
        # type narrowing
        assert self._offshore_barriers is not None
        assert self._offshore_mask is not None
        assert self._landfall_mask is not None

        # Load land layer
        logger.info('Loading land barriers "%s" from %s', h5_layer_name,
                    land_h5)
        with rex.Resource(land_h5) as res:
            if h5_layer_name not in res:
                raise ValueError(f'Dataset {h5_layer_name} not found in '
                                 '{land_h5}')
            land_profile_json = res.attrs[h5_layer_name]['profile']
            old_land_data = res[h5_layer_name][0]
        old_land_profile = json.loads(land_profile_json)

        # Reproject land barriers to new offshore projection. Set default
        # value of 1 (barrier) to cells outside of land raster.
        logger.info('Reprojecting land barriers')
        combo_barriers = self._reproject(old_land_data, old_land_profile,
                                         dtype='float32', init_dest=1)
        assert self._offshore_barriers.shape == combo_barriers.shape

        # Include offshore barriers
        mask = self._offshore_mask
        combo_barriers[mask] = self._offshore_barriers[mask]

        # Using land barriers for landfall cells. No change is needed.

        if save_tiff:
            logger.info('Saving offshore barriers combined with %s to %s',
                        h5_layer_name, self.COMBO_BARRIERS_FNAME)
            self._save_tiff(combo_barriers, self.COMBO_BARRIERS_FNAME)

        # Write to h5
        logger.info('Writing offshore barriers combined with land "%s" to '
                    '%s', h5_layer_name, offshore_h5)
        combo_barriers = combo_barriers[np.newaxis, ...]

        self._write_to_h5(offshore_h5, h5_layer_name, combo_barriers)

    def _write_to_h5(self, h5_file: str, layer_name: str, data: npt.NDArray):
        """
        Write data to an H5 file

        Parameters
        ----------
        h5_file
            Name and path of H5 file
        layer_name
            Layer name
        data
            Layer data to save
        """
        profile = self.profile()
        profile['crs'] = profile['crs'].to_proj4()
        profile['dtype'] = str(data.dtype)

        with h5py.File(h5_file, 'a') as f:
            if layer_name in f.keys():
                dset = f[layer_name]
                dset[...] = data
            else:
                dset = f.create_dataset(layer_name, data=data,
                                        chunks=H5_CHUNKS)
                dset.attrs['chunks'] = H5_CHUNKS

            dset.attrs['profile'] = json.dumps(profile)
            dset.attrs['shape'] = data.shape

    def _reproject(self, src_raster: npt.NDArray, src_profile: dict,
                   dtype: npt.DTypeLike = 'float32', init_dest: float = -1
                   ) -> npt.NDArray:
        """
        Reproject a raster into the offshore raster projection and transform.

        Parameters
        ----------
        src_raster
            Source raster
        src_profile
            Source raster profile
        dtype, optional
            Data type for destination raster, by default 'float32'
        init_dest, optional
            Value for cells outside of boundary of src_raster

        Returns
        -------
            Source data reprojected into the offshore projection.
        """
        dest_raster = np.ones(self._os_shape, dtype=dtype)
        reproject(src_raster,
                  destination=dest_raster,
                  src_transform=src_profile['transform'],
                  src_crs=src_profile['crs'],
                  dst_transform=self.profile()['transform'],
                  dst_crs=self.profile()['crs'],
                  dst_resolution=self._os_shape, num_threads=5,
                  resampling=Resampling.nearest,
                  INIT_DEST=init_dest)
        return dest_raster
