"""
Create cost multiplier raster for transmission line least cost paths

Mike Bannister
4/2/2021
"""
import numpy as np
import rasterio as rio

# Mapping NLCD land codes to land use types
NLCD_LAND_USE_CLASSES = {
    'cropland': [80, 81],
    'forest': [41, 42, 43],
    'wetland': [90, 95],
    'suburban': [21, 22, 23],
    'urban': [24],
}

DEFAULT_HILL_MULT = 1
DEFAULT_MTN_MULT = 1
DEFAULT_HILL_SLOPE = 2
DEFAULT_MTN_SLOPE = 8


class CostMultiplier:
    """
    Create cost multiplier raster for transmission line least cost paths
    """
    def __init__(self, land_use_classes=None, hill_mult=DEFAULT_HILL_MULT,
                 mtn_mult=DEFAULT_MTN_MULT, hill_slope=DEFAULT_HILL_SLOPE,
                 mtn_slope=DEFAULT_MTN_SLOPE):
        """
        Parameters
        ----------
        land_use_classes : dict
            Which NLCD values represent which land classes, e.g.
            {'forest': [41, 42, 43], 'urban': [24]}. Values must be lists.
        hill_mult : float
            Default multiplier for hilly terrain
        mtn_mult : float
            Default multiplier for mountainous terrain
        hill_slope : float
            Lowest slope in % that is considered hillly
        mtn_slope : float
            Lowest slope in % that is considered mountainous
        """
        if land_use_classes is None:
            land_use_classes = NLCD_LAND_USE_CLASSES

        self._land_use_classes = land_use_classes
        self._hill_mult = hill_mult
        self._mtn_mult = mtn_mult
        self._hill_slope = hill_slope
        self._mtn_slope = mtn_slope

    def _create_slope_mult(self, arr, config=None):
        """
        Create slope multiplier raster. Unspecified slopes are left at 1.0

        Parameters
        ----------
        arr : numpy.ndarray
            Slope raster - "Terrain slope in % of grade"
        config : dict
            Multipliers and slope cut offs for hilly and mountain terrain
        config['hill_mult'] : float
            Multiplier for hilly terrain
        config['mtn_slope'] : float
            Multiplier for mountainous terrain
        config['hill_slope'] : float
            Slope at and above which a cell is classified as hilly.
        config['mtn_slope'] : float
            Slope at and above which a cell is classified as mountainous

        Returns
        -------
        numpy.ndarray
            Multiplier raster. Minimum value for any cell is 1.
        """
        if config is None:
            config = {}
        hill_mult = config.get('hill_mult', self._hill_mult)
        mtn_mult = config.get('mtn_mult', self._mtn_mult)
        hill_slope = config.get('hill_slope', self._hill_slope)
        mtn_slope = config.get('mtn_slope', self._mtn_slope)

        hilly = (arr >= hill_slope) & (arr < mtn_slope)
        mountainous = arr >= mtn_slope

        mult_raster = np.ones(arr.shape, dtype=np.float32)
        mult_raster[hilly] = hill_mult
        mult_raster[mountainous] = mtn_mult

        return mult_raster

    def _create_land_use_mult(self, arr, multipliers):
        """
        Convert NLCD raster to land use multiplier raster. Land classes without
        specified multipliers are left a 1.

        Parameters
        ----------
        arr : numpy.ndarray
            NLCD land user raster
        multipliers : dict
            Multiplier for for land classes, E.g. {'forest': 1.5}

        Returns
        -------
        numpy.ndarray
            Multiplier raster. Minimum value for any cell is 1.
        """
        mult_raster = np.ones(arr.shape, dtype=np.float32)

        indices = []  # [(index0, multiplier0, _class_value0), ...]
        for _class, multiplier in multipliers.items():
            if _class not in self._land_use_classes:
                msg = f'Class {_class} not in self._land_use_classes: ' +\
                      f'{self._land_use_classes}'
                raise ValueError(msg)

            values = self._land_use_classes[_class]
            assert isinstance(values, list)

            for value in values:
                index = arr == value
                indices.append((index, multiplier, value))

        for i in indices:
            mult_raster[i[0]] = i[1]

        return mult_raster

    def create_mults_raster(self, iso_regions, land_use, slope, iso_config,
                            default):
        """
        Create x-mission line cost multiplier raster

        Parameters
        ----------
        iso_regions : numpy.ndarray(int)
            Raster defining ISO regions
        land_use : numpy.ndarray(int)
            Land use raster using NLCD codes
        slope : numpy.ndarray(float)
            Slope raster as % grade
        iso_config : list of dict
            Multiplier definitions for iso regions, example format:
            [
                {
                    'iso': 1,
                    'land_use': {'forest': 3, 'wetland': 6},
                    'slope': {'hill_mult': 2, 'mtn_mult': 4, 'hill_slope': 25,
                            'mtn_slope': 50}
                }, ...
            ]
        default : dict
            Multiplier definitions for regions not defined in iso_config, e.g.:
            {
                'land_use': {'forest': 3, 'wetland': 6},
                'slope': {'hill_mult': 2, 'mtn_mult': 4, 'hill_slope': 25,
                        'mtn_slope': 50}
            }

        Returns
        -------
        numpy.ndarray
            Multiplier raster
        """
        assert iso_regions.shape == land_use.shape == slope.shape, \
            'All arrays must be the same shape'

        mults_arr = np.ones(iso_regions.shape, dtype=np.float32)
        regions_mask = np.full(mults_arr.shape, False, dtype=bool)

        for r_conf in iso_config:
            print('Processing region', r_conf['iso'])
            mask = iso_regions == r_conf['iso']
            regions_mask = regions_mask | mask

            if 'land_use' in r_conf:
                r_lu = land_use[mask]
                lu_mult = self._create_land_use_mult(r_lu, r_conf['land_use'])
                mults_arr[mask] = lu_mult

            if 'slope' in r_conf:
                r_slope = slope[mask]
                slope_mult = self._create_slope_mult(r_slope, r_conf['slope'])
                mults_arr[mask] = mults_arr[mask] * slope_mult

        # Calculate multipliers for regions not defined in `config`
        print('Processing default region')
        default_mask = ~regions_mask

        if 'land_use' in default:
            rlu = land_use[default_mask]
            lu_mult = self._create_land_use_mult(rlu, default['land_use'])
            mults_arr[default_mask] = lu_mult

        if 'slope' in default:
            r_slope = slope[default_mask]
            slope_mult = self._create_slope_mult(r_slope, default['slope'])
            mults_arr[default_mask] = mults_arr[default_mask] * slope_mult

        return mults_arr

    # TODO - this run() is redundant, should we save mults raster, or just
    # keep it in memeory? Actually should save it as it only needs to be
    # calculated once, then can be used for parallel processing
    @classmethod
    def run(cls, iso_regions, land_use, slope, iso_config, default):
        """
        Create x-mission line cost multiplier raster

        Parameters
        ----------
        iso_regions : numpy.ndarray(int)
            Raster defining ISO regions
        land_use : numpy.ndarray(int)
            Land use raster using NLCD codes
        slope : numpy.ndarray(float)
            Slope raster as % grade
        iso_config : list of dict
            Multiplier definitions for iso regions
        default : dict
            Multiplier definitions for regions not defined in iso_config

        Returns
        -------
        numpy.ndarray
            Multiplier raster
        """
        cm = cls()
        mults = cm.create_mults_raster(iso_regions, land_use, slope,
                                       iso_config, default)
        return mults

    # TODO - this is nice, but current processing pipeline is using npy
    @staticmethod
    def save_geotiff(mults, template, outf):
        """
        Save multiplier raster to geotiff

        Parameters
        ----------
        mults : numpy.array
            Multipliers array
        template : str
            Filename for CONUS template raster
        outf : str
            Filename for geotiff
        """
        ras = rio.open(template)
        ras_out = rio.open(outf,
                           'w',
                           driver='GTiff',
                           height=ras.shape[0],
                           width=ras.shape[1],
                           count=1,
                           dtype=mults.dtype,
                           crs=ras.crs,
                           transform=ras.transform,
                           compress='lzw'
                           )
        ras_out.write(mults, 1)
        ras_out.close()
