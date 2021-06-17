"""
Functions for loading substations, transmission line, etc

Mike Bannister
5/18/2021
"""
import os
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio as rio

from .config import TEMPLATE_SHAPE, power_classes, power_to_voltage, \
    BARRIERS_MULT
from .utilities import RowColTransformer

logger = logging.getLogger(__name__)


class LoadData:
    """
    Load data from disk
    """
    def __init__(self, capacity_class, resolution=128,
                 costs_dir='cost_rasters',
                 template_f='data/conus_template.tif',
                 landuse_f='data/nlcd.npy',
                 slope_f='data/slope.npy',
                 barriers_f='data/transmission_barriers.tif',
                 sc_points_f='data/sc_points/fips_run_agg_new.csv',
                 # all_conns_f='/home/mbannist/conus_allconns.gpkg',
                 all_conns_f='data/conus_allconns.gpkg',
                 iso_regions_f='data/iso_regions.tif'):
        """
        Parameters
        ----------
        capacity_class : String
            Desired reV power capacity class, one of "100MW", "200MW", "400MW",
            "1000MW"
        resolution : Int
            resolution is CURRENTLY IGNORED
            Desired Supply Curve Point resolution, one of: 32, 64, 128
        TODO
       exclusions : numpy.ndarray(int) | None
            Exclusions layer. Any cell that is > 0 will have a final
            multiplier of -1, which blocks it from least cost paths.
        """
        assert capacity_class in power_classes.keys(), 'capacity must be ' + \
            f'one of {list(power_classes.keys())}'

        assert resolution == 128, 'Only resolutions of 128 are currently ' + \
            'supported'

        self.capacity_class = capacity_class
        self.rct = RowColTransformer(template_f)

        with rio.open(template_f) as ras:
            self.crs = ras.profile['crs']

        # Real world power capacity (MW)
        self.tie_power = power_classes[capacity_class]

        # Voltage (kV) corresponding to self.tie_power
        self.tie_voltage = power_to_voltage[str(self.tie_power)]

        logger.debug(f'Loading transmission features from {all_conns_f}')
        cl = AllConnsLoader(all_conns_f)
        self.subs = cl.subs[cl.subs.max_volts >= self.tie_voltage]
        self.t_lines = cl.t_lines[cl.t_lines.voltage >= self.tie_voltage]
        self.lcs = cl.lcs
        self.sinks = cl.sinks

        logger.debug('Loading rasters')
        costs_f = os.path.join(costs_dir, f'costs_{self.tie_power}MW.tif')
        logger.debug(f'Loading costs from {costs_f}')
        self.costs_arr = load_raster(costs_f)
        assert self.costs_arr.min() > 0, 'All costs must have a positive value'

        self.regions_arr = load_raster(iso_regions_f)
        self._barriers_arr = load_raster(barriers_f)
        assert self.costs_arr.shape == self.regions_arr.shape == \
            self._barriers_arr.shape, 'All rasters must have the same shape'

        self._barriers_arr[self._barriers_arr == 1] = BARRIERS_MULT
        self._barriers_arr[self._barriers_arr == 0] = 1
        self.paths_arr = self.costs_arr * self._barriers_arr

        self._plot_costs_arr = None

        logger.debug(f'Loading SC points from {sc_points_f}')
        # TODO - make this resolution aware
        self.sc_points = self._load_sc_points(sc_points_f)

    def _load_sc_points(self, sc_points_f, raw_crs='epsg:4326'):
        """
        Load supply curve points from disk

        Parameters
        ----------
        sc_points_f : String
            Path to supply curve points CSV
        raw_crs : String
            CRS string for SC points file

        Returns
        -------
        sc_points : List of SupplyCurvePoint
        """
        pts = pd.read_csv(sc_points_f)
        geo = [Point(xy) for xy in zip(pts.longitude, pts.latitude)]
        pts = gpd.GeoDataFrame(pts, crs=raw_crs, geometry=geo).to_crs(self.crs)

        sc_points = []
        for _, row in pts.iterrows():
            sc_pt = SupplyCurvePoint(row.sc_point_gid, row.sc_row_ind,
                                     row.sc_col_ind, row.geometry, self.rct,
                                     self.regions_arr)
            sc_points.append(sc_pt)
        return sc_points

    @property
    def plot_costs_arr(self):
        """
        Return costs array for plotting with transmission barriers set to -1
        """
        if self._plot_costs_arr is None:
            self._plot_costs_arr = self.costs_arr.copy()
            self._plot_costs_arr[self._barriers_arr == BARRIERS_MULT] = -1

        return self._plot_costs_arr


class AllConnsLoader:
    def __init__(self, all_conns_f, subs_f='_substations_cache.shp'):
        """
        Load substations, transmission lines, load centers, and sinks from disc

        Parameters
        ----------
        all_conns_f : String
            Path to conus_allcons.gpkg
        subs_f : String
            Path to substations shapefile that is autogenerated by this class
        """
        conns = gpd.read_file(all_conns_f)
        conns = conns.drop(['bgid', 'egid', 'cap_left'], axis=1)

        self.t_lines = conns[conns.category == 'TransLine']
        self.lcs = conns[conns.category == 'LoadCen']
        self.sinks = conns[conns.category == 'PCALoadCen']

        subs_f = os.path.join(os.path.dirname(all_conns_f), subs_f)
        if os.path.exists(subs_f):
            logger.debug(f'Loading cached substations from {subs_f}')
            self.subs = gpd.read_file(subs_f)
        else:
            self.subs = conns[conns.category == 'Substation']
            self._update_sub_volts()
            self.subs.to_file(subs_f)

    def _update_sub_volts(self):
        """
        Get substation voltages from trans lines
        """
        logger.debug('Determining voltages for substations, this will take '
                     'a while')
        self.subs['temp_volts'] = self.subs.apply(self._get_volts, axis=1)
        volts = self.subs.temp_volts.str.split('/', expand=True)
        self.subs[['min_volts', 'max_volts']] = volts
        self.subs.min_volts = self.subs.min_volts.astype(np.int16)
        self.subs.max_volts = self.subs.max_volts.astype(np.int16)
        self.subs.drop(['voltage', 'temp_volts'], axis=1, inplace=True)

    def _get_volts(self, row):
        """
        Determine min/max volts for substation from trans lines

        Parameters
        ----------
        row : pandas.DataFrame row
            Row being processed

        Returns
        -------
        str
            min/max connected volts, e.g. "69/250" (kV)
        """
        tl_ids = [int(x) for x in row.trans_gids[1:-1].split(',')]
        lines = self.t_lines[self.t_lines.gid.isin(tl_ids)]
        volts = lines.voltage.values
        if len(volts) == 0:
            msg = ('No transmission lines found connected to substation '
                   f'{row.gid}. Setting voltage to 0')
            logger.warning(msg)
            volts = [0]
        return f'{int(min(volts))}/{int(max(volts))}'


class SupplyCurvePoint:
    def __init__(self, id, sc_row_ind, sc_col_ind, geo, rct, regions_arr):
        """
        Represents a supply curve point for possible renewable energy plant.

        Parameters
        ----------
        id : int
            Id of supply curve point
        sc_row_ind : int
            Row of SC point in grid by resolution
        sc_col_ind : int
            Column of SC point in grid by resolution
        geo : shapely.geometry.Point
            Point projected to template raster
        rct : RowColTransformer
            Transformer for template raster
        regions_arr : numpy.ndarray
            ISO regions raster
        """
        self.id = id
        self.sc_row_ind = sc_row_ind
        self.sc_col_ind = sc_col_ind
        self.x = geo.x
        self.y = geo.y

        # Calculate and save location on template raster
        row, col = rct.get_row_col(self.x, self.y)
        self.row = row
        self.col = col

        self.region = regions_arr[row, col]
        self.point = geo

    def __repr__(self):
        return f'id={self.id}, ' +\
               f'sc_ind=({self.sc_row_ind}, {self.sc_col_ind}),' +\
               f'coords=({self.x}, {self.y}), ' +\
               f'r/c=({self.row}, {self.col})'


def load_raster(f_name):
    """
    Load raster in same shape as template from disc.

    Parameters
    ----------
    f_name : String
        Path and name of raster

    Returns
    -------
    data : numpy.ndarray
    """
    _, ext = os.path.splitext(f_name)

    if ext == '.tif' or ext == '.tiff':
        with rio.open(f_name) as dataset:
            data = dataset.read(1)

    elif ext == '.npy':
        data = np.load(f_name)
        if len(data.shape) == 3:
            data = data[0]

    else:
        raise ValueError(f'Unknown extension type on {f_name}')

    assert data.shape == TEMPLATE_SHAPE
    return data
