"""
Region Classifier Module
- Used to classify meta points with a label from a shapefile

Sample Usage:
```
meta_path = 'meta.csv'
regions_path = 'us_states.shp'
save_to = 'new_meta.csv'
regions_label = 'NAME'

classifier = region_classifier(meta_path=meta_path, regions_path=regions_path,
                               regions_label=regions_label)
classification = classifier.classify(save_to=save_to)
```
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import shape
from scipy.spatial import cKDTree
import logging

logger = logging.getLogger(__name__)


class region_classifier():
    """ Base class of region classification """

    CRS = {'init': 'epsg:4326'}
    DEFAULT_REGIONS_LABEL = 'regions_index'

    def __init__(self, meta_path, regions_path, regions_label=None,
                 lat_label="LATITUDE", long_label="LONGITUDE"):
        """
        Parameters
        ----------
        meta_path : str
            Path to meta CSV file containing lat/lon points
        regions_path : str
            Path to regions shapefile containing labeled geometries
        regions_label : str
            Attribute to use a label in the regions shapefile
        lat_label : str
            Latitude column name in meta CSV file
        long_label : str
            Longitude column name in meta CSV file
        """
        self.meta_path = meta_path
        self.regions_path = regions_path
        self.lat_label = lat_label
        self.long_label = long_label
        self.regions_label = regions_label

        self.meta = self.get_meta()
        self.regions = self.get_regions()

    def get_regions(self):
        """ Load the regions shapefile into geopandas dataframe """
        regions = gpd.read_file(self.regions_path).to_crs(self.CRS)
        if self.regions_label not in regions.columns:
            self.regions_label = self.DEFAULT_REGIONS_LABEL
            regions[self.regions_label] = regions.index
            logger.warning('Setting regions label: ' + str(self.regions_label))
        centroids = regions.geometry.centroid
        regions[self.long_label] = centroids.x
        regions[self.lat_label] = centroids.y
        return regions

    def get_meta(self):
        """ Load the meta csv file into geopandas dataframe """
        meta = pd.read_csv(self.meta_path)
        geometry = [Point(xy) for xy in zip(meta[self.long_label],
                                            meta[self.lat_label])]
        meta = gpd.GeoDataFrame(meta, crs=self.CRS, geometry=geometry)
        return meta

    def classify(self, save_to=None):
        """ Classify the meta data with regions labels
        Parameters
        ----------
        save_to : str
            Optional output path for labeled meta CSV file
        """
        try:
            # Get intersection classifications
            meta_inds, region_inds, outlier_inds = self.intersect()
        except Exception as e:
            invalid_geom_ids = self.check_geometry()
            if len(invalid_geom_ids > 0):
                logger.error('The following geometries are invalid:')
                logger.error(invalid_geom_ids)
            else:
                raise e

        # Check for any intersection outliers
        if len(outlier_inds) > 0:
            # Lookup the nearest region geometry (by centroid)
            logger.warning('The following points are outliers:')
            logger.warning(outlier_inds)
            cols = [self.lat_label, self.long_label]
            lookup = self.regions[cols]
            target = self.meta.loc[outlier_inds][cols]
            meta_inds += outlier_inds
            region_inds += list(self.nearest(target, lookup))

        region_labels = self.regions.loc[region_inds, self.regions_label]
        # Build classification mapping
        data = np.array([meta_inds, region_labels]).T
        classified = pd.DataFrame(data=data, columns=['meta_index',
                                                      self.regions_label])
        classified.set_index('meta_index', inplace=True)

        classified_meta = self.meta.loc[meta_inds]
        classified_meta[self.regions_label] = classified[self.regions_label]

        # Output
        if save_to:
            output_meta = classified_meta.drop('geometry', axis=1)
            output_meta.to_csv(save_to, index=False)
        return classified_meta

    def intersect(self):
        """ Join the meta points to regions by spatial intersection """

        joined = gpd.sjoin(self.meta, self.regions,
                           how='inner', op='intersects')
        if 'index_left' in joined.columns:
            joined = joined.drop_duplicates('index_left', keep='last')
            meta_inds = list(joined['index_left'])
        else:
            meta_inds = list(joined.index)
        region_inds = list(joined['index_right'])
        outliers = self.meta.loc[~self.meta.index.isin(meta_inds)]
        outlier_inds = list(outliers.index)
        return meta_inds, region_inds, outlier_inds

    @staticmethod
    def nearest(target, lookup):
        """ Lookup the indices to the nearest point
        Parameters
        ----------
        target : Pandas DataFrame
            List of lat/lon points
        lookup : Pandas DataFrame
            List of lat/lon points
        """
        tree = cKDTree(lookup)
        _, inds = tree.query(target, k=1)
        return inds

    @staticmethod
    def geom_is_valid(geom):
        """ Check if individual geometry is valid """
        try:
            shape(geom)
            return 1
        except AttributeError:
            return 0

    def check_geometry(self):
        """ Get index list of invalid geometries """
        isvalid = self.regions.geometry.apply(lambda x: self.geom_is_valid(x))
        return list(self.regions[isvalid == 0].index)
