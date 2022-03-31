# -*- coding: utf-8 -*-
"""
Compute setbacks exclusions
"""
import logging
import os
import geopandas as gpd

from rex.utilities import log_mem

from reVX.setbacks.base import BaseSetbacks


logger = logging.getLogger(__name__)


class ParcelSetbacks(BaseSetbacks):
    """
    Parcel setbacks, using negative buffers.
    """

    def compute_generic_setbacks(self, features_fpath):
        """Compute generic setbacks.

        Parameters
        ----------
        features_fpath : str
            Path to shape file with features to compute setbacks from.

        Returns
        -------
        setbacks : ndarray
            Raster array of setbacks
        """
        logger.info('Computing generic setbacks')
        setback_features = self._parse_features(features_fpath)

        setbacks = [
            (geom, 1) for geom in setback_features.buffer(0).difference(
                setback_features.buffer(-1 * self.generic_setback)
            )
        ]

        return self._rasterize_setbacks(setbacks)

    @staticmethod
    def _compute_local_setbacks(features, cnty, setback):
        """Compute local features setbacks.

        Parameters
        ----------
        features : geopandas.GeoDataFrame
            Features to setback from.
        cnty : geopandas.GeoDataFrame
            Regulations for a single county.
        setback : int
            Setback distance in meters.

        Returns
        -------
        setbacks : list
            List of setback geometries.
        """
        logger.debug('- Computing setbacks for county FIPS {}'
                     .format(cnty.iloc[0]['FIPS']))
        log_mem(logger)
        mask = features.centroid.within(cnty['geometry'].values[0])
        tmp = features.loc[mask]

        setbacks = [
            (geom, 1) for geom in tmp.buffer(0).difference(
                tmp.buffer(-1 * setback)
            )
        ]

        return setbacks

    def _parse_regulations(self, regulations_fpath):
        """
        Parse parcel regulations, reduce table to just property lines

        Parameters
        ----------
        regulations_fpath : str
            Path to parcel regulations .csv file

        Returns
        -------
        regulations : pandas.DataFrame
            Parcel regulations table
        """
        regulations = super()._parse_regulations(regulations_fpath)

        mask = regulations['Feature Type'].apply(str.strip) == 'Property Line'
        regulations = regulations.loc[mask]

        return regulations

    def _check_regulations(self, features_fpath):
        """
        Reduce regs to state corresponding to features_fpath if needed.

        Parameters
        ----------
        features_fpath : str
            Path to shape file with features to compute setbacks from.

        Returns
        -------
        regulations : geopandas.GeoDataFrame
            Parcel regulations
        """
        state = os.path.basename(features_fpath).split('.')[0]
        state = ''.join(filter(str.isalpha, state.lower()))

        regulation_states = self.regulations.State.apply(
            lambda s: ''.join(filter(str.isalpha, s.lower()))
        )

        mask = regulation_states == state
        regulations = self.regulations[mask].reset_index(drop=True)

        logger.debug(
            'Computing setbacks for parcel regulations in {} counties'
            .format(len(regulations))
        )

        return regulations

    def _parse_features(self, features_fpath):
        """Abstract method to parse features.

        Parameters
        ----------
        features_fpath : str
            Path to file containing features to setback from.

        Returns
        -------
        geopandas.GeoDataFrame
            Geometries of features to setback from in exclusion coordinate
            system
        """
        features = gpd.read_file(features_fpath)
        if features.crs is None:
            features = features.set_crs("EPSG:4326")
        return features.to_crs(crs=self.crs)

    @classmethod
    def run(cls, excl_fpath, parcels_path, out_dir, plant_height,
            regulations_fpath=None, multiplier=None,
            chunks=(128, 128), max_workers=None, replace=False, hsds=False):
        """
        Compute parcel setbacks and write them to a geotiff.
        If a regulations file is given, compute local setbacks, otherwise
        compute generic setbacks using the given multiplier and the plant
        height.

        Parameters
        ----------
        excl_fpath : str
            Path to .h5 file containing exclusion layers, will also be the
            location of any new setback layers
        parcels_path : str
            Path to parcels file or directory containing parcel files.
            This path can contain any pattern that can be used in the glob
            function. For example, `/path/to/features/[A]*` would match
            with all the features in the direcotry
            `/path/to/features/` that start with "A". This input
            can also be a directory, but that directory must ONLY
            contain feature files. If your feature files are mixed
            with other files or directories, use something like
            `/path/to/features/*.geojson`.
        out_dir : str
            Directory to save setbacks geotiff(s) into
        plant_height : float | int
            Plant height (m), used to determine setback distance using
            multiplier.
        regulations_fpath : str | None, optional
            Path to regulations .csv file, if None create generic
            setbacks using plant height * "multiplier", by default None
        multiplier : int | float | str | None, optional
            Setback multiplier to use if regulations are not supplied.
            It is multiplied with plant height to calculate the setback
            distance. If supplied along with `regulations_fpath`, this input
            will be ignored, by default None.
        chunks : tuple, optional
            Chunk size to use for setback layers, if None use default chunk
            size in excl_fpath, by default (128, 128)
        max_workers : int, optional
            Number of workers to use for setback computation, if 1 run in
            serial, if > 1 run in parallel with that many workers, if None
            run in parallel on all available cores, by default None
        replace : bool, optional
            Flag to replace geotiff if it already exists, by default False
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        """
        setbacks = cls(excl_fpath, plant_height=plant_height,
                       regulations_fpath=regulations_fpath,
                       multiplier=multiplier,
                       hsds=hsds, chunks=chunks)

        parcels_path = setbacks._get_feature_paths(parcels_path)
        for fpath in parcels_path:
            geotiff = os.path.basename(fpath).split('.')[0]
            geotiff += '.tif'
            geotiff = os.path.join(out_dir, geotiff)
            if os.path.exists(geotiff) and not replace:
                msg = ('{} already exists, setbacks will not be re-computed '
                       'unless replace=True'.format(geotiff))
                logger.error(msg)
            else:
                logger.info("Computing setbacks from parcels in {} and saving "
                            "to {}".format(fpath, geotiff))
                setbacks.compute_setbacks(fpath, geotiff=geotiff,
                                          max_workers=max_workers,
                                          replace=replace)
