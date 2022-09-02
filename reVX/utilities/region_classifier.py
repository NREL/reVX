"""
Region Classifier Module
"""
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import shape
from scipy.spatial import cKDTree
import logging

from reVX.utilities.utilities import log_versions
from rex import Resource

logger = logging.getLogger(__name__)


class RegionClassifier():
    """
    Base class of region classification

    Examples
    --------
    >>> meta_path = 'meta.csv'
    >>> regions_path = 'us_states.shp'
    >>> regions_label = 'NAME'
    >>>
    >>> classifier = RegionClassifier(meta_path=meta_path,
                                      regions_path=regions_path,
                                      regions_label=regions_label)
    >>>
    >>> force = True
    >>> fout = 'new_meta.csv'
    >>>
    >>> classification = classifier.classify(force=force)
    >>> classifier.output_to_csv(classification, fout)
    """

    CRS = "EPSG:4326"
    DEFAULT_REGIONS_LABEL = 'regions_index'

    def __init__(self, meta_path, regions, regions_label=None,
                 outlier_value=-999):
        """
        Parameters
        ----------
        meta_path : str | pandas.DataFrame
            Path to meta CSV file, resource .h5 file, or pre-loaded meta
            DataFrame containing lat/lon points
        regions : str | GeoDataFrame
            Path to regions shapefile containing labeled geometries or
            a pre-loaded GeoDataFrame
        regions_label : str
            Attribute to use as label in the regions shapefile
        outlier_value : float | int | str
            Value to assign to outliers if not force
        """
        log_versions(logger)
        self._regions_label = regions_label
        if self._regions_label is None:
            self._regions_label = self.DEFAULT_REGIONS_LABEL

        self._meta = self._get_meta(meta_path)
        self._regions = self._get_regions(regions, self._regions_label)
        self._outlier_value = outlier_value

    @property
    def regions(self):
        """Get the regions GeoDataFrame

        Returns
        -------
        GeoDataFrame
        """
        return self._regions

    @staticmethod
    def output_to_csv(gdf, path):
        """ Export a geopandas dataframe to csv

        Parameters
        ----------
        gdf : GeoPandas DataFrame
            Meta data to export
        path : str
            Output CSV file path for labeled meta CSV file
        """
        output_gdf = gdf.drop('geometry', axis=1)
        if output_gdf.index.name == 'gid':
            output_gdf = output_gdf.reset_index()

        output_gdf.to_csv(path, index=False)

    @classmethod
    def _get_regions(cls, regions, regions_label):
        """ Load the regions shapefile into geopandas dataframe

        Parameters
        ----------
        regions : str | GeoDataFrame
            Path to regions shapefile containing labeled geometries or
            a pre-loaded GeoDataFrame
        regions_label : str
            Attribute to use as label in the regions shapefile
        """

        if not isinstance(regions, gpd.GeoDataFrame):
            regions = gpd.read_file(regions).to_crs(cls.CRS)
            if regions_label not in regions.columns:
                regions_label = cls.DEFAULT_REGIONS_LABEL
                regions[regions_label] = regions.index
                logger.warning('Setting regions label: {}'
                               .format(regions_label))

        centroids = regions.geometry.centroid
        regions['longitude'] = centroids.x
        regions['latitude'] = centroids.y

        return regions

    @classmethod
    def _get_meta(cls, meta_path):
        """ Load the meta csv file into geopandas dataframe

        Parameters
        ----------
        meta_path : str | pandas.DataFrame
            Path to meta CSV file, resource .h5 file, or pre-loaded meta
            DataFrame containing lat/lon points
        """
        if isinstance(meta_path, str):
            if meta_path.endswith('.csv'):
                meta = pd.read_csv(meta_path)
            elif meta_path.endswith('.h5'):
                with Resource(meta_path) as f:
                    meta = f.meta
            else:
                msg = ("Cannot parse meta data from {}, expecting a .csv or "
                       ".h5 file!".format(meta_path))
                logger.error(msg)
                raise RuntimeError(msg)
        elif isinstance(meta_path, pd.DataFrame):
            meta = meta_path
        else:
            msg = ("Cannot parse meta data from {}, expecting a .csv or "
                   ".h5 file path, or a pre-loaded pandas DataFrame"
                   .format(meta_path))
            logger.error(msg)
            raise RuntimeError(msg)

        lat_label, long_label = cls._get_lat_lon_labels(meta)
        geometry = [Point(xy) for xy in zip(meta[long_label],
                                            meta[lat_label])]
        meta = gpd.GeoDataFrame(meta, crs=cls.CRS,
                                geometry=geometry)

        return meta

    @staticmethod
    def _get_lat_lon_labels(df):
        """ Auto detect the latitude and longitude columns from DataFrame

        Parameters
        ----------
        df : Pandas DataFrame
            Meta data with the latitude/longitude columns to detect
        """

        # Latitude
        lat_col = [c for c in df if c.lower().startswith('lat')]
        if len(lat_col) > 1:
            msg = "Multiple latitude columns found: {}".format(lat_col)
            logger.error(msg)
            raise RuntimeError(msg)

        lat_col = lat_col[0]

        # Longitude
        lon_col = [c for c in df if c.lower().startswith('lon')]
        if len(lon_col) > 1:
            msg = "Multiple longitude columns found: {}".format(lon_col)
            logger.error(msg)
            raise RuntimeError(msg)

        lon_col = lon_col[0]

        return [lat_col, lon_col]

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
        tree = cKDTree(lookup)  # pylint: disable=not-callable
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

    def classify(self, force=False):
        """ Classify the meta data with regions labels

        Parameters
        ----------
        force : str
            Force outlier classification by finding nearest
        """
        try:
            # Get intersection classifications
            meta_inds, region_inds, outlier_inds = self._intersect()
        except Exception as e:
            logger.warning(e)
            invalid_geom_ids = self._check_geometry()
            if invalid_geom_ids:
                logger.warning('The following geometries are invalid: {}'
                               .format(invalid_geom_ids))
            else:
                logger.exception('Cannot run region classification')
                raise

        classified_meta = self._meta.copy()
        classified_meta[self._regions_label] = self._outlier_value
        region_labels = self._regions.loc[region_inds, self._regions_label]
        region_labels = list(region_labels)
        classified_meta.loc[meta_inds, self._regions_label] = region_labels

        # Check for any intersection outliers
        num_outliers = len(outlier_inds)
        if (num_outliers and force):
            # Lookup the nearest region geometry (by centroid)
            logger.warning('The following points are outliers:')
            logger.warning(outlier_inds)
            cols = self._get_lat_lon_labels(self._meta)
            lookup = self._regions[['latitude', 'longitude']]
            target = self._meta.loc[outlier_inds][cols]
            out_inds = list(self._nearest(target, lookup))

            regions = self._regions.loc[out_inds, self._regions_label]
            regions = list(regions)
            classified_meta.loc[outlier_inds, self._regions_label] = regions

        return classified_meta

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

    def _check_geometry(self):
        """ Get index list of invalid geometries """
        geometry = self._regions.geometry
        isvalid = geometry.apply(self._geom_is_valid)

        return list(self._regions[isvalid == 0].index)

    @classmethod
    def run(cls, meta_path, regions, regions_label=None,
            force=False, fout=None):
        """ Run full classification

        Parameters
        ----------
        meta_path : str | pandas.DataFrame
            Path to meta CSV file, resource .h5 file, or pre-loaded meta
            DataFrame containing lat/lon points
        regions : str | GeoDataFrame
            Path to regions shapefile containing labeled geometries or
            a pre-loaded GeoDataFrame
        regions_label : str
            Attribute to use a label in the regions shapefile
        force : str
            Force outlier classification by finding nearest
        fout : str
            Output CSV file path for labeled meta CSV file

        Examples
        --------
        >>> meta_path = 'meta.csv'
        >>> regions_path = 'us_states.shp'
        >>> regions_label = 'NAME'
        >>> force = True
        >>> fout = 'new_meta.csv'
        >>>
        >>> RegionClassifier.run(meta_path=meta_path,
                                 regions_path=regions_path,
                                 regions_label=regions_label
                                 force=force, fout=fout)
        """
        classifier = cls(meta_path=meta_path, regions=regions,
                         regions_label=regions_label)
        classification = classifier.classify(force=force)
        if fout:
            cls.output_to_csv(classification, fout)

        return classification
