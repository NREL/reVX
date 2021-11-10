# -*- coding: utf-8 -*-
"""
Compute the mean wind direction for each supply curve point
"""
import logging
import numpy as np

from reV.supply_curve.aggregation import Aggregation, AggFileHandler
from reV.supply_curve.extent import SupplyCurveExtent
from reV.utilities.exceptions import EmptySupplyCurvePointError
from reVX.wind_dirs.mean_wind_dirs_point import MeanWindDirectionsPoint
from reVX.utilities.utilities import log_versions
from rex.utilities.loggers import log_mem

logger = logging.getLogger(__name__)


class MeanWindDirections(Aggregation):
    """
    Average the wind direction via the wind vectors.
    Then convert to equivalent sc_point_gid
    """

    def __init__(self, res_h5_fpath, excl_fpath, wdir_dsets,
                 tm_dset='techmap_wtk', excl_dict=None,
                 area_filter_kernel='queen', min_area=None,
                 resolution=128, excl_area=None):
        """
        Parameters
        ----------
        res_h5_fpath : str
            Filepath to .h5 file containing wind direction data
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        wdir_dsets : str | list
            Wind direction dataset to average
        tm_dset : str, optional
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data,
            by default 'techmap_wtk'
        excl_dict : dict | None, optional
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
            by default None
        area_filter_kernel : str, optional
            Contiguous area filter method to use on final exclusions mask,
            by default 'queen'
        min_area : float | None, optional
            Minimum required contiguous area filter in sq-km, by default None
        resolution : int | None, optional
            SC resolution, must be input in combination with gid,
            by default 128
        excl_area : float | None, optional
            Area of an exclusion pixel in km2. None will try to infer the area
            from the profile transform attribute in excl_fpath,
            by default None
        """
        log_versions(logger)
        if isinstance(wdir_dsets, str):
            wdir_dsets = [wdir_dsets]

        for dset in wdir_dsets:
            if not dset.startswith('winddirection'):
                msg = ('{} is not a valid wind direction dataset!'
                       .format(dset))
                logger.error(msg)
                raise ValueError(msg)

        super().__init__(excl_fpath, res_h5_fpath, tm_dset, *wdir_dsets,
                         excl_dict=excl_dict,
                         area_filter_kernel=area_filter_kernel,
                         min_area=min_area,
                         resolution=resolution, excl_area=excl_area)

    # pylint: disable=unused-argument
    @classmethod
    def run_serial(cls, excl_fpath, h5_fpath, tm_dset, *wind_dir_dset,
                   excl_dict=None, inclusion_mask=None,
                   area_filter_kernel='queen', min_area=None,
                   resolution=128, excl_area=0.0081, gids=None,
                   gen_index=None, **kwargs):
        """
        Standalone method to aggregate - can be parallelized.

        Parameters
        ----------
        excl_fpath : str | list | tuple
            Filepath to exclusions h5 with techmap dataset
            (can be one or more filepaths).
        h5_fpath : str
            Filepath to .h5 file to aggregate
        tm_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        wind_dir_dset : str
            Wind directions to aggreate, can supply multiple datasets
        excl_dict : dict, optional
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
            by default None
        area_filter_kernel : str, optional
            Contiguous area filter method to use on final exclusions mask,
            by default "queen"
        min_area : float, optional
            Minimum required contiguous area filter in sq-km,
            by default None
        resolution : int, optional
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead,
            by default 0.0081
        excl_area : float, optional
            Area of an exclusion pixel in km2. None will try to infer the area
            from the profile transform attribute in excl_fpath,
            by default None
        gids : list, optional
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent, by default None
        gen_index : np.ndarray, optional
            Array of generation gids with array index equal to resource gid.
            Array value is -1 if the resource index was not used in the
            generation run, by default None
        kwargs : dict
            Unused kwargs from Aggregation.run_serial method, namely agg_method

        Returns
        -------
        agg_out : dict
            Aggregated values for each aggregation dataset
        """
        with SupplyCurveExtent(excl_fpath, resolution=resolution) as sc:
            exclusion_shape = sc.exclusions.shape
            if gids is None:
                gids = sc.valid_sc_points(tm_dset)
            elif np.issubdtype(type(gids), np.number):
                gids = [gids]

            slice_lookup = sc.get_slice_lookup(gids)

        cls._check_inclusion_mask(inclusion_mask, gids, exclusion_shape)

        # pre-extract handlers so they are not repeatedly initialized
        file_kwargs = {'excl_dict': excl_dict,
                       'area_filter_kernel': area_filter_kernel,
                       'min_area': min_area}
        dsets = wind_dir_dset + ('meta', )
        agg_out = {ds: [] for ds in dsets}
        with AggFileHandler(excl_fpath, h5_fpath, **file_kwargs) as fh:
            n_finished = 0
            for gid in gids:
                gid_inclusions = cls._get_gid_inclusion_mask(
                    inclusion_mask, gid, slice_lookup,
                    resolution=resolution)
                try:
                    gid_out = MeanWindDirectionsPoint.run(
                        gid,
                        fh.exclusions,
                        fh.h5,
                        tm_dset,
                        *wind_dir_dset,
                        excl_dict=excl_dict,
                        inclusion_mask=gid_inclusions,
                        resolution=resolution,
                        excl_area=excl_area,
                        exclusion_shape=exclusion_shape,
                        close=False,
                        gen_index=gen_index)

                except EmptySupplyCurvePointError:
                    logger.debug('SC gid {} is fully excluded or does not '
                                 'have any valid source data!'.format(gid))
                except Exception:
                    logger.exception('SC gid {} failed!'.format(gid))
                    raise
                else:
                    n_finished += 1
                    logger.debug('Serial aggregation: '
                                 '{} out of {} points complete'
                                 .format(n_finished, len(gids)))
                    log_mem(logger)
                    for k, v in gid_out.items():
                        agg_out[k].append(v)

        return agg_out

    def aggregate(self, max_workers=None, sites_per_worker=1000):
        """
        Average wind directions to sc_points

        Parameters
        ----------
        max_workers : int | None
            Number of cores to run summary on. None is all
            available cpus.
        sites_per_worker : int, optional
            Number of SC points to process on a single parallel worker,
            by default 1000

        Returns
        -------
        agg : dict
            Aggregated values for each aggregation dataset
        """
        agg = super().aggregate(max_workers=max_workers,
                                sites_per_worker=sites_per_worker)

        return agg

    @classmethod
    def run(cls, res_h5_fpath, excl_fpath, wdir_dsets,
            tm_dset='techmap_wtk', excl_dict=None,
            area_filter_kernel='queen', min_area=None,
            resolution=128, excl_area=None, max_workers=None,
            sites_per_worker=1000, out_fpath=None):
        """
        Aggregate powerrose to supply curve points, find neighboring supply
        curve point gids and rank them based on prominent powerrose direction

        Parameters
        ----------
        res_h5_fpath : str
            Filepath to .h5 file containing wind direction data
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        wdir_dsets : str | list
            Wind direction dataset to average
        tm_dset : str, optional
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data,
            by default 'techmap_wtk'
        excl_dict : dict | None, optional
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
            by default None
        area_filter_kernel : str, optional
            Contiguous area filter method to use on final exclusions mask,
            by default 'queen'
        min_area : float | None, optional
            Minimum required contiguous area filter in sq-km, by default None
        resolution : int | None, optional
            SC resolution, must be input in combination with gid,
            by default 128
        excl_area : float | None, optional
            Area of an exclusion pixel in km2. None will try to infer the area
            from the profile transform attribute in excl_fpath,
            by default None
        max_workers : int | None, optional
            Number of cores to run summary on. None is all
            available cpus, by default None
        sites_per_worker : int, optional
            Number of SC points to process on a single parallel worker,
            by default 1000
        out_fpath : str
            Path to .h5 file to save aggregated data too

        Returns
        -------
        agg : dict
            Aggregated values for each aggregation dataset
        """
        wdir = cls(res_h5_fpath, excl_fpath, wdir_dsets, tm_dset=tm_dset,
                   excl_dict=excl_dict, area_filter_kernel=area_filter_kernel,
                   min_area=min_area, resolution=resolution,
                   excl_area=excl_area)

        agg = wdir.aggregate(max_workers=max_workers,
                             sites_per_worker=sites_per_worker)

        if out_fpath is not None:
            wdir.save_agg_to_h5(out_fpath, agg)

        return agg
