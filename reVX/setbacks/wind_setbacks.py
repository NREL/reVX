# -*- coding: utf-8 -*-
"""
Compute setbacks exclusions
"""
import os
import re
import logging
from warnings import warn
import fiona
import geopandas as gpd

from rex.utilities import log_mem
from reVX.setbacks.base import BaseSetbacks

logger = logging.getLogger(__name__)


class BaseWindSetbacks(BaseSetbacks):
    """
    Create exclusions layers for wind setbacks
    """

    MULTIPLIERS = {'high': 3, 'moderate': 1.1}

    def __init__(self, excl_fpath, hub_height, rotor_diameter,
                 regulations_fpath=None, multiplier=None, hsds=False,
                 chunks=(128, 128)):
        """
        Parameters
        ----------
        excl_fpath : str
            Path to .h5 file containing exclusion layers, will also be
            the location of any new setback layers.
        hub_height : float | int
            Turbine hub height (m), used along with rotor diameter to
            compute blade tip height which is used to determine setback
            distance.
        rotor_diameter : float | int
            Turbine rotor diameter (m), used along with hub height to
            compute blade tip height which is used to determine setback
            distance.
        regulations_fpath : str | None, optional
            Path to regulations .csv file. At a minimum, this csv must
            contain the following columns: `Value Type`, which
            specifies wether the value is a multiplier or static height,
            `Value`, which specifies the numeric value of the setback or
            multiplier, and `FIPS`, which specifies a unique 5-digit
            code for each county (this can be an integer - no leading
            zeros required). Typically, this csv will also have a
            `Feature Type` column that labels the type of setback
            that each row represents. Valid options for the `Value Type`
            are:
                - "Max-tip Height Multiplier"
                - "Rotor-Diameter Multiplier"
                - "Hub-height Multiplier"
                - "Meters"
            If this input is `None`, a generic setback of
            `max_tip_height * multiplier` is used. By default `None`.
        multiplier : int | float | str | None, optional
            Setback multiplier to use if wind regulations are not
            supplied. It is multiplied with max-tip height to calculate
            the setback distance. If str, must be a one of
            {'high', 'moderate'}. If supplied along with
            `regulations_fpath`, this input will be
            ignored. By default `None`.
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on
            AWS behind HSDS. By default `False`.
        chunks : tuple, optional
            Chunk size to use for setback layers, if None use default
            chunk size in excl_fpath. By default `(128, 128)`.
        """
        self._hub_height = hub_height
        self._rotor_diameter = rotor_diameter
        super().__init__(
            excl_fpath=excl_fpath,
            base_setback_dist=(
                self._hub_height + self._rotor_diameter / 2
            ),
            regulations_fpath=regulations_fpath, multiplier=multiplier,
            hsds=hsds, chunks=chunks
        )

    @property
    def hub_height(self):
        """
        Turbine hub-height in meters

        Returns
        -------
        float
        """
        return self._hub_height

    @property
    def rotor_diameter(self):
        """
        Turbine rotor diameter in meters

        Returns
        -------
        float
        """
        return self._rotor_diameter

    def get_regulation_setback(self, county_regulations):
        """
        Compute the setback distance in meters from the county
        regulations, turbine tip height or rotor diameter.

        Parameters
        ----------
        county_regulations : pandas.Series
            Pandas Series with wind regulations for a single county or
            feature type. At a minimum, this Series must
            contain the following columns: `Value Type`, which
            specifies wether the value is a multiplier or static height,
            `Value`, which specifies the numeric value of the setback or
            multiplier. Valid options for the `Value Type` are:
                - "Max-tip Height Multiplier"
                - "Rotor-Diameter Multiplier"
                - "Hub-height Multiplier"
                - "Meters"

        Returns
        -------
        setback : float | None
            setback distance in meters, None if the setback "Value Type"
            was not recognized
        """

        setback_type = county_regulations["Value Type"]
        setback = county_regulations["Value"]
        if setback_type.lower() == "max-tip height multiplier":
            setback *= self.base_setback_dist
        elif setback_type.lower() == "rotor-diameter multiplier":
            setback *= self.rotor_diameter
        elif setback_type.lower() == "hub-height multiplier":
            setback *= self.hub_height
        elif setback_type.lower() != "meters":
            msg = ("Cannot create setback for {}, expecting "
                   '"Max-tip Height Multiplier", '
                   '"Rotor-Diameter Multiplier", '
                   '"Hub-height Multiplier", or '
                   '"Meters", but got {}'
                   .format(county_regulations["County"], setback_type))
            logger.warning(msg)
            warn(msg)
            setback = None

        return setback

    def _preflight_check(self, regulations_fpath, multiplier):
        """
        Run preflight checks on WindSetBack inputs:
        1) Ensure either a wind regulations .csv is provided, or
           a setback multiplier.
        2) Ensure wind regulations has county FIPS, map regulations to
           county geometries from exclusions .h5 file.
        3) Ensure multiplier is a valid entry, either a float or one of
           {'high', 'moderate'}

        Parameters
        ----------
        regulations_fpath : str | None
            Path to wind regulations .csv file, if None create global
            setbacks.
        multiplier : int | float | str | None
            setback multiplier to use if wind regulations are not
            supplied, if str, must one of {'high', 'moderate'}.

        Returns
        -------
        regulations: `geopandas.GeoDataFrame` | None
            GeoDataFrame with county level wind setback regulations
            merged with county geometries, use for intersecting with
            setback features.
        Multiplier : float | None
            Generic setbacks multiplier.
        """
        regulations, multiplier = super()._preflight_check(
            regulations_fpath, multiplier
        )
        if isinstance(multiplier, str):
            multiplier = self.MULTIPLIERS[multiplier]
            logger.debug('Computing setbacks using generic Max-tip Height '
                         'Multiplier of {}'.format(multiplier))
        return regulations, multiplier


class StructureWindSetbacks(BaseWindSetbacks):
    """
    Structure Wind setbacks
    """

    @staticmethod
    def _split_state_name(state_name):
        """
        Split state name at capitals to map .geojson files to
        regulations state names.

        Parameters
        ----------
        state_name : str
            State name from geojson files paths with out spaces

        Returns
        -------
        str
            State names with spaces added between Capitals (names) to
            match wind regulations state names.
        """
        state_name = ' '.join(a for a
                              in re.split(r'([A-Z][a-z]*)', state_name)
                              if a)

        return state_name

    @staticmethod
    def _get_feature_paths(features_fpath):
        """
        Find all structures .geojson files in structures dir

        Parameters
        ----------
        features_fpath : str
            Path to structures geojson for a single state, or directory
            containing geojsons for all states. Used to identify
            structures to build setbacks from. Files should be by state.

        Returns
        -------
        file_paths : list
            List of file paths to all structures .geojson files in
            structures_dir
        """
        if features_fpath.endswith('.geojson'):
            file_paths = [features_fpath]
        else:
            file_paths = []
            for file in sorted(os.listdir(features_fpath)):
                if file.endswith('.geojson'):
                    file_paths.append(os.path.join(features_fpath, file))

        return file_paths

    def _parse_regulations(self, regulations_fpath):
        """
        Parse wind regulations, reduce table to just structures

        Parameters
        ----------
        regulations_fpath : str
            Path to wind regulations .csv file

        Returns
        -------
        regulations : pandas.DataFrame
            Wind regulations table
        """
        regulations = super()._parse_regulations(regulations_fpath)

        mask = ((regulations['Feature Type'] == 'Structures')
                & (regulations['Comment'] != 'Occupied Community Buildings'))
        regulations = regulations.loc[mask]

        return regulations

    def _check_regulations(self, features_fpath):
        """
        Reduce regs to state corresponding to features_fpath if needed

        Parameters
        ----------
        features_fpath : str
            Path to shape file with features to compute setbacks from

        Returns
        -------
        wind_regs : geopandas.GeoDataFrame | None
            Wind Regulations
        """
        state_name = os.path.basename(features_fpath).split('.')[0]
        state = self._split_state_name(state_name)
        wind_regs = self.regulations
        mask = wind_regs["State"] == state

        if not mask.any():
            msg = ("There are no local regulations in {}!".format(state))
            logger.error(msg)
            raise RuntimeError(msg)

        wind_regs = wind_regs.loc[mask].reset_index(drop=True)
        logger.debug('Computing setbacks for wind regulations in {} counties'
                     .format(len(wind_regs)))

        return wind_regs

    @classmethod
    def run(cls, excl_fpath, structures_path, out_dir, hub_height,
            rotor_diameter, regulations_fpath=None, multiplier=None,
            chunks=(128, 128), max_workers=None, replace=False, hsds=False):
        """
        Compute state's structural setbacks and write them to a geotiff.
        If a wind regulations file is given compute local setbacks,
        otherwise compute generic setbacks using the given multiplier
        and the turbine tip-height.

        Parameters
        ----------
        excl_fpath : str
            Path to .h5 file containing exclusion layers, will also be
            the location of any new setback layers
        structure_path : str
            Path to structures geojson for a single state, or directory
            containing geojsons for all states.
        out_dir : str
            Directory to save setbacks geotiff(s) into
        hub_height : float | int
            Turbine hub height (m), used along with rotor diameter to
            compute blade tip height which is used to determine setback
            distance.
        rotor_diameter : float | int
            Turbine rotor diameter (m), used along with hub height to
            compute blade tip height which is used to determine setback
            distance.
        regulations_fpath : str | None, optional
            Path to regulations .csv file. At a minimum, this csv must
            contain the following columns: `Value Type`, which
            specifies wether the value is a multiplier or static height,
            `Value`, which specifies the numeric value of the setback or
            multiplier, and `FIPS`, which specifies a unique 5-digit
            code for each county (this can be an integer - no leading
            zeros required). Typically, this csv will also have a
            `Feature Type` column that labels the type of setback
            that each row represents. Valid options for the `Value Type`
            are:
                - "Max-tip Height Multiplier"
                - "Rotor-Diameter Multiplier"
                - "Hub-height Multiplier"
                - "Meters"
            If this input is `None`, a generic setback of
            `max_tip_height * multiplier` is used. By default `None`.
        multiplier : int | float | str | None, optional
            Setback multiplier to use if wind regulations are not
            supplied.It is multiplied with max-tip height to calculate
            the setback distance. If str, must be one of
            {'high', 'moderate'}. If supplied along with
            `regulations_fpath`, this input will be ignored.
            By default `None`.
        chunks : tuple, optional
            Chunk size to use for setback layers, if None use default
            chunk size in excl_fpath. By default `(128, 128)`.
        max_workers : int, optional
            Number of workers to use for setback computation, if 1 run
            in serial, if > 1 run in parallel with that many workers,
            if `None`, run in parallel on all available cores.
            By default `None`.
        replace : bool, optional
            Flag to replace geotiff if it already exists.
            By default `False`.
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on
            AWS behind HSDS. By default `False`.
        """
        setbacks = cls(excl_fpath, hub_height, rotor_diameter,
                       regulations_fpath=regulations_fpath,
                       multiplier=multiplier,
                       hsds=hsds,
                       chunks=chunks)

        structures_path = setbacks._get_feature_paths(structures_path)

        for fpath in structures_path:
            geotiff = os.path.basename(fpath).replace('.geojson', '.tif')
            geotiff = os.path.join(out_dir, geotiff)
            if os.path.exists(geotiff) and not replace:
                msg = ('{} already exists, setbacks will not be re-computed '
                       'unless replace=True'.format(geotiff))
                logger.error(msg)
            else:
                logger.info("Computing setbacks from structures in {} and "
                            "saving to {}".format(fpath, geotiff))
                setbacks.compute_setbacks(fpath, geotiff=geotiff,
                                          max_workers=max_workers,
                                          replace=replace)


class RoadWindSetbacks(BaseWindSetbacks):
    """
    Road Wind setbacks
    """

    def _parse_features(self, features_fpath):
        """
        Load roads from gdb file, convert to exclusions coordinate
        system.

        Parameters
        ----------
        features_fpath : str
            Path to here streets gdb file for given state.

        Returns
        -------
        roads : `geopandas.GeoDataFrame.sindex`
            Geometries for roads in gdb file, in exclusion coordinate
            system
        """
        lyr = fiona.listlayers(features_fpath)[0]
        roads = gpd.read_file(features_fpath, driver='FileGDB', layer=lyr)

        return roads.to_crs(crs=self.crs)

    @staticmethod
    def _get_feature_paths(features_fpath):
        """
        Find all roads gdb files in roads_dir

        Parameters
        ----------
        features_fpath : str
            Path to state here streets gdb file or directory containing
            states gdb files. Used to identify roads to build setbacks
            from. Files should be by state.

        Returns
        -------
        file_paths : list
            List of file paths to all roads .gdp files in roads_dir
        """
        if features_fpath.endswith('.gdb'):
            file_paths = [features_fpath]
        else:
            file_paths = []
            for file in sorted(os.listdir(features_fpath)):
                if file.endswith('.gdb') and file.startswith('Streets_USA'):
                    file_paths.append(os.path.join(features_fpath, file))

        return file_paths

    def _parse_regulations(self, regulations_fpath):
        """
        Parse wind regulations, reduce table to just roads

        Parameters
        ----------
        regulations_fpath : str
            Path to wind regulations .csv file

        Returns
        -------
        regulations : pandas.DataFrame
            Wind regulations table
        """
        regulations = super()._parse_regulations(regulations_fpath)

        feature_types = {'Roads', 'Highways', 'Highways 111'}
        mask = regulations['Feature Type'].isin(feature_types)
        regulations = regulations.loc[mask]

        return regulations

    def _check_regulations(self, features_fpath):
        """
        Reduce regs to state corresponding to features_fpath if needed

        Parameters
        ----------
        features_fpath : str
            Path to shape file with features to compute setbacks from

        Returns
        -------
        wind_regs : geopandas.GeoDataFrame | None
            Wind Regulations
        """
        state = features_fpath.split('.')[0].split('_')[-1]
        wind_regs = self.regulations
        mask = wind_regs['Abbr'] == state

        if not mask.any():
            msg = ("There are no local regulations in {}!".format(state))
            logger.error(msg)
            raise RuntimeError(msg)

        wind_regs = wind_regs.loc[mask].reset_index(drop=True)
        logger.debug('Computing setbacks for wind regulations in {} counties'
                     .format(len(wind_regs)))

        return wind_regs

    @classmethod
    def run(cls, excl_fpath, roads_path, out_dir, hub_height,
            rotor_diameter, regulations_fpath=None, multiplier=None,
            chunks=(128, 128), max_workers=None, replace=False, hsds=False):
        """
        Compute state's road setbacks and write them to a geotiff.
        If a wind regulations file is given compute local setbacks,
        otherwise compute generic setbacks using the given multiplier
        and the turbine tip-height.

        Parameters
        ----------
        excl_fpath : str
            Path to .h5 file containing exclusion layers, will also be
            the location of any new setback layers.
        road_path : str
            Path to state here streets gdb file or directory containing
            states gdb files.
        out_dir : str
            Directory to save setbacks geotiff(s) into
        hub_height : float | int
            Turbine hub height (m), used along with rotor diameter to
            compute blade tip height which is used to determine setback
            distance.
        rotor_diameter : float | int
            Turbine rotor diameter (m), used along with hub height to
            compute blade tip height which is used to determine setback
            distance.
        regulations_fpath : str | None, optional
            Path to regulations .csv file. At a minimum, this csv must
            contain the following columns: `Value Type`, which
            specifies wether the value is a multiplier or static height,
            `Value`, which specifies the numeric value of the setback or
            multiplier, and `FIPS`, which specifies a unique 5-digit
            code for each county (this can be an integer - no leading
            zeros required). Typically, this csv will also have a
            `Feature Type` column that labels the type of setback
            that each row represents. Valid options for the `Value Type`
            are:
                - "Max-tip Height Multiplier"
                - "Rotor-Diameter Multiplier"
                - "Hub-height Multiplier"
                - "Meters"
            If this input is `None`, a generic setback of
            `max_tip_height * multiplier` is used. By default `None`.
        multiplier : int | float | str | None, optional
            Setback multiplier to use if wind regulations are not
            supplied. It is multiplied with max-tip height to calculate
            the setback distance. If str, must be one of
            {'high', 'moderate'}. If supplied along with
            `regulations_fpath`, this input will be ignored.
            By default `None`.
        chunks : tuple, optional
            Chunk size to use for setback layers, if None use default
            chunk size in excl_fpath. By default `(128, 128)`.
        max_workers : int, optional
            Number of workers to use for setback computation, if 1 run
            in serial, if > 1 run in parallel with that many workers,
            if `None`, run in parallel on all available cores.
            By default `None`.
        replace : bool, optional
            Flag to replace geotiff if it already exists.
            By default `False`.
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on
            AWS behind HSDS. By default `False`.
        """
        setbacks = cls(excl_fpath, hub_height, rotor_diameter,
                       regulations_fpath=regulations_fpath,
                       multiplier=multiplier,
                       hsds=hsds, chunks=chunks)

        roads_path = setbacks._get_feature_paths(roads_path)
        for fpath in roads_path:
            geotiff = os.path.basename(fpath).replace('.gdb', '.tif')
            geotiff = os.path.join(out_dir, geotiff)
            if os.path.exists(geotiff) and not replace:
                msg = ('{} already exists, setbacks will not be re-computed '
                       'unless replace=True'.format(geotiff))
                logger.error(msg)
            else:
                logger.info("Computing setbacks from roads in {} and saving "
                            "to {}".format(fpath, geotiff))
                setbacks.compute_setbacks(fpath, geotiff=geotiff,
                                          max_workers=max_workers,
                                          replace=replace)


class TransmissionWindSetbacks(BaseWindSetbacks):
    """
    Transmission Wind setbacks, computed against a single set of
    transmission features instead of against state level features.
    """

    @staticmethod
    def _compute_local_setbacks(features, cnty, setback):
        """Compute local county setbacks.

        This method will compute the setbacks using a county-specific
        regulations file that specifies either a static setback or a
        multiplier value that will be used along with plant height
        specifications to compute the setback.

        Parameters
        ----------
        features : geopandas.GeoDataFrame
            Features to setback from
        cnty : geopandas.GeoDataFrame
            Wind regulations for a single county
        setback : int
            Setback distance in meters

        Returns
        -------
        setbacks : list
            List of setback geometries
        """
        logger.debug('- Computing setbacks for county FIPS {}'
                     .format(cnty.iloc[0]['FIPS']))
        log_mem(logger)
        tmp = gpd.clip(features, cnty)
        tmp = tmp[~tmp.is_empty]

        # Buffer setback
        tmp.loc[:, 'geometry'] = tmp.buffer(setback)

        setbacks = [(geom, 1) for geom in tmp['geometry']]

        return setbacks

    def _parse_regulations(self, regulations_fpath):
        """
        Parse wind regulations, reduce table to just transmission

        Parameters
        ----------
        regulations_fpath : str
            Path to wind regulations .csv file

        Returns
        -------
        regulations : pandas.DataFrame
            Wind regulations table
        """
        regulations = super()._parse_regulations(regulations_fpath)

        mask = regulations['Feature Type'] == 'Transmission'
        regulations = regulations.loc[mask]

        return regulations

    @classmethod
    def run(cls, excl_fpath, features_fpath, out_dir, hub_height,
            rotor_diameter, regulations_fpath=None, multiplier=None,
            chunks=(128, 128), max_workers=None, replace=False, hsds=False):
        """
        Compute setbacks from given features and write them to a
        geotiff. If a wind regulations file is given compute local
        setbacks, otherwise compute generic setbacks using the given
        multiplier and the turbine tip-height.

        Parameters
        ----------
        excl_fpath : str
            Path to .h5 file containing exclusion layers, will also be
            the location of any new setback layers.
        features_fpath : str
            Path to shape file with transmission or rail features to
            compute setbacks from.
        out_dir : str
            Directory to save geotiff containing rasterized setbacks
            into.
        hub_height : float | int
            Turbine hub height (m), used along with rotor diameter to
            compute blade tip height which is used to determine setback
            distance.
        rotor_diameter : float | int
            Turbine rotor diameter (m), used along with hub height to
            compute blade tip height which is used to determine setback
            distance.
        regulations_fpath : str | None, optional
            Path to regulations .csv file. At a minimum, this csv must
            contain the following columns: `Value Type`, which
            specifies wether the value is a multiplier or static height,
            `Value`, which specifies the numeric value of the setback or
            multiplier, and `FIPS`, which specifies a unique 5-digit
            code for each county (this can be an integer - no leading
            zeros required). Typically, this csv will also have a
            `Feature Type` column that labels the type of setback
            that each row represents. Valid options for the `Value Type`
            are:
                - "Max-tip Height Multiplier"
                - "Rotor-Diameter Multiplier"
                - "Hub-height Multiplier"
                - "Meters"
            If this input is `None`, a generic setback of
            `max_tip_height * multiplier` is used. By default `None`.
        multiplier : int | float | str | None, optional
            Setback multiplier to use if wind regulations are not
            supplied. It is multiplied with max-tip height to calculate
            the setback distance. If str, must be one of
            {'high', 'moderate'}. If supplied along with
            `regulations_fpath`, this input will be ignored.
            By default `None`.
        chunks : tuple, optional
            Chunk size to use for setback layers, if None use default
            chunk size in excl_fpath. By default `(128, 128)`.
        max_workers : int, optional
            Number of workers to use for setback computation, if 1 run
            in serial, if > 1 run in parallel with that many workers,
            if `None`, run in parallel on all available cores.
            By default `None`.
        replace : bool, optional
            Flag to replace geotiff if it already exists.
            By default `False`.
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on
            AWS behind HSDS. By default `False`.
        """
        geotiff = os.path.basename(features_fpath).split('.')[0]
        geotiff += '.tif'
        geotiff = os.path.join(out_dir, geotiff)
        if os.path.exists(geotiff) and not replace:
            msg = ('{} already exists, setbacks will not be re-computed '
                   'unless replace=True'.format(geotiff))
            logger.error(msg)
        else:
            setbacks = cls(excl_fpath, hub_height, rotor_diameter,
                           regulations_fpath=regulations_fpath,
                           multiplier=multiplier,
                           hsds=hsds, chunks=chunks)

            logger.info("Computing setbacks from {} and saving "
                        "to {}".format(features_fpath, geotiff))
            setbacks.compute_setbacks(features_fpath, geotiff=geotiff,
                                      max_workers=max_workers,
                                      replace=replace)


class RailWindSetbacks(TransmissionWindSetbacks):
    """Rail Wind setbacks.

    These setbacks are computed against a single set of railroad
    features instead of state level features. This class uses the same
    computational approach as `TransmissionWindSetbacks`.
    """

    def _parse_regulations(self, regulations_fpath):
        """
        Parse wind regulations, reduce table to just rail

        Parameters
        ----------
        regulations_fpath : str
            Path to wind regulations .csv file.

        Returns
        -------
        regulations : `pandas.DataFrame`
            Wind regulations table.
        """
        # pylint: disable=bad-super-call
        sup = super(TransmissionWindSetbacks, self)
        regulations = sup._parse_regulations(regulations_fpath)

        mask = regulations['Feature Type'] == 'Railroads'
        regulations = regulations.loc[mask]

        return regulations
