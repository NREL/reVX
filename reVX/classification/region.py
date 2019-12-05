"""
Region Classifier Module


Sample Usage:
```
meta_path = 'meta.csv'
regions_path = 'us_states.shp'
save_to = 'new_meta.csv'

classifier = region_classifier(meta_path=meta_path, regions_path=regions_path)
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

    CRS = {'init': 'epsg:4326'}
    DEFAULT_META_LABEL = 'meta_index'
    DEFAULT_REGIONS_LABEL = 'regions_index'

    def __init__(self, meta_path, regions_path,
                 meta_label=None, regions_label=None,
                 lat_label="LATITUDE", long_label="LONGITUDE"):
        self.meta_path = meta_path
        self.regions_path = regions_path
        self.lat_label = lat_label
        self.long_label = long_label
        self.meta_label = meta_label
        self.regions_label = regions_label

        self.meta = self.get_meta()
        self.regions = self.get_regions()

    def get_regions(self):
        """ """
        regions = gpd.read_file(self.regions_path).to_crs(self.CRS)
        if self.regions_label not in regions.columns:
            self.regions_label = self.DEFAULT_REGIONS_LABEL
            regions[self.regions_label] = regions.index
            logger.warning('Setting meta label: ' + str(self.regions_label))
        regions.set_index(self.regions_label, inplace=True, drop=False)
        centroids = regions.geometry.centroid
        regions[self.long_label] = centroids.x
        regions[self.lat_label] = centroids.y
        return regions

    def get_meta(self):
        """ """
        meta = pd.read_csv(self.meta_path)
        if self.meta_label not in meta.columns:
            self.meta_label = self.DEFAULT_META_LABEL
            meta[self.meta_label] = meta.index
            logger.warning('Setting regions label: ' + str(self.meta_label))
        geometry = [Point(xy) for xy in zip(meta[self.long_label],
                                            meta[self.lat_label])]
        meta = gpd.GeoDataFrame(meta, crs=self.CRS, geometry=geometry)
        meta.set_index(self.meta_label, inplace=True, drop=False)
        return meta

    def classify(self, save_to=None):
        """ """
        # Get intersection classifications
        try:
            meta_ids, region_ids, outlier_ids = self.intersect()
        except Exception as e:
            logger.error(e)
            invalid_geom_ids = self.check_geometry()
            logger.error('The following geometries are invalid:')
            logger.error(invalid_geom_ids)

        # Handle the intersection outliers
        if len(outlier_ids) > 0:
            logger.warning('The following points are outliers:')
            logger.warning(outlier_ids)
            cols = [self.lat_label, self.long_label]

            # Lookup the nearest assigned points
            lookup = self.regions[cols]
            target = self.meta.loc[outlier_ids][cols]
            inds = self.nearest(target, lookup)

            meta_ids += outlier_ids
            region_ids += list(self.regions.loc[inds, self.regions_label])

        # Build classification mapping
        data = np.array([meta_ids, region_ids]).T
        classified = pd.DataFrame(data=data, columns=[self.meta_label,
                                                      self.regions_label])
        classified.sort_values(self.meta_label, inplace=True)
        classified.set_index(self.meta_label, inplace=True)

        classified_meta = self.meta.loc[meta_ids].sort_values(self.meta_label)
        classified_meta[self.regions_label] = classified[self.regions_label]

        # Output
        if save_to:
            del classified_meta['geometry']
            classified_meta.to_csv(save_to, index=False)
        return classified_meta

    def intersect(self):
        """ """

        joined = gpd.sjoin(self.meta, self.regions,
                           how='inner', op='intersects')
        joined = joined.drop_duplicates(self.meta_label, keep='last')
        meta_ids = list(joined[self.meta_label])
        region_ids = list(joined[self.regions_label])
        outliers = self.meta.loc[~self.meta[self.meta_label].isin(meta_ids)]
        outlier_ids = list(outliers[self.meta_label])
        return meta_ids, region_ids, outlier_ids

    @staticmethod
    def nearest(target, lookup):
        """ """
        tree = cKDTree(lookup)
        _, inds = tree.query(target, k=1)
        return inds

    @staticmethod
    def geom_is_valid(geom):
        """ """
        try:
            shape(geom)
            return 1
        except AttributeError:
            return 0

    @classmethod
    def check_geometry(self):
        """ """
        isvalid = self.regions.geometry.apply(lambda x: self.geom_is_valid(x))
        return list(self.regions[isvalid == 0].index)
