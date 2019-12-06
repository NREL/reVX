"""
Region Classifier Module
- Used to classify meta points with a label from a shapefile

Sample Usage:
```
meta_path = 'meta.csv'
regions_path = 'us_states.shp'
regions_label = 'NAME'
lat_label = 'LATITUDE'
long_label = 'LONGITUDE'

classifier = region_classifier(meta_path=meta_path, regions_path=regions_path,
                               lat_label=lat_label, long_label=long_label,
                               regions_label=regions_label)

save_to = 'new_meta.csv'
force = True

classification = classifier.classify(save_to=save_to, force=force)
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

    def __init__(self, meta_path, regions_path, lat_label, long_label,
                 regions_label=None):
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
        self._meta_path = meta_path
        self._regions_path = regions_path
        self._lat_label = lat_label
        self._long_label = long_label
        self._regions_label = regions_label

        self._meta = self.get_meta()
        self._regions = self.get_regions()

    def get_regions(self):
        """ Load the regions shapefile into geopandas dataframe """
        regions = gpd.read_file(self._regions_path).to_crs(self.CRS)
        if self._regions_label not in regions.columns:
            self._regions_label = self.DEFAULT_REGIONS_LABEL
            regions[self._regions_label] = regions.index
            logger.warning('Setting regions label: ' + self._regions_label)
        centroids = regions.geometry.centroid
        regions[self._long_label] = centroids.x
        regions[self._lat_label] = centroids.y
        return regions

    def get_meta(self):
        """ Load the meta csv file into geopandas dataframe """
        meta = pd.read_csv(self._meta_path)
        geometry = [Point(xy) for xy in zip(meta[self._long_label],
                                            meta[self._lat_label])]
        meta = gpd.GeoDataFrame(meta, crs=self.CRS, geometry=geometry)
        return meta

    def classify(self, save_to=None, force=False):
        """ Classify the meta data with regions labels
        Parameters
        ----------
        save_to : str
            Optional output path for labeled meta CSV file
        force : str
            Force outlier classification by finding nearest
        """
        try:
            # Get intersection classifications
            meta_inds, region_inds, outlier_inds = self._intersect()
        except Exception as e:
            invalid_geom_ids = self._check_geometry()
            if invalid_geom_ids:
                logger.error('The following geometries are invalid:')
                logger.error(invalid_geom_ids)
            else:
                raise e

        # Check for any intersection outliers
        num_outliers = len(outlier_inds)
        if ((num_outliers > 0) & (force is True)):
            # Lookup the nearest region geometry (by centroid)
            logger.warning('The following points are outliers:')
            logger.warning(outlier_inds)
            cols = [self._lat_label, self._long_label]
            lookup = self._regions[cols]
            target = self._meta.loc[outlier_inds][cols]
            region_inds += list(self._nearest(target, lookup))

        # Get full list of meta indices and region labels
        meta_inds += outlier_inds
        region_labels = list(self._regions.loc[region_inds,
                                               self._regions_label])
        if ((num_outliers > 0) & (force is False)):
            # Fill unclassified labels
            region_labels += [-999 for _ in range(num_outliers)]

        # Build classification mapping
        region_labels = np.array(region_labels).astype(str)
        classified_meta = self._meta.loc[meta_inds]
        classified_meta[self._regions_label] = region_labels
        classified_meta.sort_index(inplace=True)

        # Output
        if save_to:
            self.output_to_csv(classified_meta, save_to)
        return classified_meta

    @staticmethod
    def output_to_csv(gdf, path):
        """ Export a geopandas dataframe to csv """
        output_gdf = gdf.drop('geometry', axis=1)
        output_gdf.to_csv(path, index=False)

    def _intersect(self):
        """ Join the meta points to regions by spatial intersection """

        joined = gpd.sjoin(self._meta, self._regions,
                           how='inner', op='intersects')
        if 'index_left' in joined.columns:
            joined = joined.drop_duplicates('index_left', keep='last')
            meta_inds = list(joined['index_left'])
        else:
            meta_inds = list(joined.index)
        region_inds = list(joined['index_right'])
        outliers = self._meta.loc[~self._meta.index.isin(meta_inds)]
        outlier_inds = list(outliers.index)
        return meta_inds, region_inds, outlier_inds

    @staticmethod
    def _nearest(target, lookup):
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
    def _geom_is_valid(geom):
        """ Check if individual geometry is valid """
        try:
            shape(geom)
            return 1
        except AttributeError:
            return 0

    def _check_geometry(self):
        """ Get index list of invalid geometries """
        geometry = self._regions.geometry
        isvalid = geometry.apply(lambda x: self._geom_is_valid(x))
        return list(self._regions[isvalid == 0].index)
