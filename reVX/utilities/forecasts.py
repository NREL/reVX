"""
Forecast Utilities
"""
import h5py
import logging
import numpy as np
import os
import shutil

from rex import Resource

logger = logging.getLogger(__name__)


class Forecasts:
    """
    Forecast Utility
    """
    def __init__(self, fcst_h5, fcst_dset, actuals_h5=None, actuals_dset=None):
        """
        Parameters
        ----------
        fcst_h5 : str
            Path to forecast .h5 file
        fcst_dset : str
            Dataset to correct
        actuals_h5 : str, optional
            Path to forecast to .h5 file, by default None
        actuals_dset : str, optional
            Actuals dataset
        """
        self._fcst_h5 = fcst_h5
        if actuals_h5 is None:
            actuals_h5 = fcst_h5

        self._actuals_h5 = actuals_h5

        self._fcst_dset = fcst_dset

        if actuals_dset is None:
            actuals_dset = fcst_dset

        self._actuals_dset = actuals_dset

        self._preflight_check()

    @property
    def fcst_h5(self):
        """
        Path to forecast .h5 file

        Returns
        -------
        str
        """
        return self._fcst_h5

    @property
    def actuals_h5(self):
        """
        Path to actuals .h5 file

        Returns
        -------
        str
        """
        return self._actuals_h5

    @property
    def fcst_dset(self):
        """
        Forecast dset to correct

        Returns
        -------
        str
        """
        return self._fcst_dset

    @property
    def actuals_dset(self):
        """
        Actuals dset to use to correct forecasts

        Returns
        -------
        str
        """
        return self._actuals_dset

    @staticmethod
    def bias_correct_fcst(actuals, fcsts):
        """
        Bias correct forecast data

        Parameters
        ----------
        actuals : ndarray
            Timeseries actuals (time x sites)
        fcsts : ndarray
            Timeseries forecats (time x sites)


        Returns
        -------
        fcsts : ndarray
            Bias corrected forecasts
        """
        bc_factors = actuals.sum(axis=0) / fcsts.sum(axis=0)
        fcsts = fcsts * bc_factors
        actuals_max = actuals.max(axis=0)
        mask = fcsts >= actuals_max
        fcsts = np.where(mask, actuals_max, fcsts)

        return fcsts

    @classmethod
    def blend_fcsts(cls, actuals, fcsts, fcst_perc):
        """
        Bias correct and blend forecasts with actuals

        Parameters
        ----------
        actuals : ndarray
            Timeseries actuals (time x sites)
        fcsts : ndarray
            Timeseries forecats (time x sites)
        fcst_perc : float
            Percentage of forecast to use for blending

        Returns
        -------
        fcsts : ndarray
            Bias corrected and blended forecasts
        """
        fcsts = cls.bias_correct_fcst(actuals, fcsts)
        fcsts = ((actuals * (1 - fcst_perc)) + (fcsts * fcst_perc))

        return fcsts

    @classmethod
    def _correct(cls, fcst, actuals, fcst_perc=None):
        """
        Correct given data

        Parameters
        ----------
        actuals : ndarray
            Timeseries actuals (time x sites)
        fcsts : ndarray
            Timeseries forecats (time x sites)
        fcst_perc : float, optional
            Percentage of forecast to use for blending, by default None

        Returns
        -------
        fcsts : ndarray
            Corrected forecasts
        """
        mae = cls.compute_mae(actuals, fcst)
        logger.debug('Forecast agg MAE: {:.4f}, ave MAE {:.4f}'
                     .format(*mae))

        if fcst_perc is not None:
            logger.info('Bias correcting and blending forecasts with '
                        '{:}% actuals'.format((1 - fcst_perc) * 100))
            fcst = cls.blend_fcsts(actuals, fcst, fcst_perc)
        else:
            logger.info('Bias correcting forecasts')
            fcst = cls.bias_correct_fcst(actuals, fcst)

        mae = cls.compute_mae(actuals, fcst)
        logger.debug('Corrected forecast agg MAE: {:.4f}, '
                     'ave MAE {:.4f}'.format(*mae))

        return fcst

    @staticmethod
    def compute_mae(actuals, fcsts):
        """
        Compute aggregate and average MAE between actuals and forecasts

        Parameters
        ----------
        actuals : ndarray
            Timeseries actuals (time x sites)
        fcsts : ndarray
            Timeseries forecats (time x sites)

        Returns
        -------
        agg_mae : float
            Aggregate MAE for all sites
        ave_mae : float
            Average MEA for all sites
        """
        agg_mae = (np.abs(np.nansum(fcsts - actuals, axis=1)).sum()
                   / np.nansum(np.nanmax(actuals, axis=0) * 8760))
        ave_mae = np.nanmean(np.abs(fcsts - actuals).sum(axis=0)
                             / (actuals.max(axis=0) * 8760))

        return agg_mae, ave_mae

    def _preflight_check(self):
        """
        Check to ensure dset is available in forecast and actuals .h5 files
        """
        with Resource(self.fcst_h5) as f:
            if self.fcst_dset not in f:
                msg = ('{} is not a valid dataset in forecast file: {}'
                       .format(self.fcst_dset, self.fcst_h5))
                logger.error(msg)
                raise RuntimeError(msg)

        with Resource(self.actuals_h5) as f:
            if self.actuals_dset not in f:
                msg = ('{} is not a valid dataset in actuals file: {}'
                       .format(self.actuals_dset, self.actuals_h5))
                logger.error(msg)
                raise RuntimeError(msg)

    def correct_dsets(self, out_h5, fcst_perc=None):
        """
        Bias correct and blend (if requested) forecasts

        Parameters
        ----------
        out_h5 : str
            Output path for corrected .h5 file
        fcst_perc : float, optional
            Percentage of forecast to use for blending, by default None
        """
        if not os.path.exists(out_h5):
            logger.debug('Copying forecasts ({}) to output path ({})'
                         .format(self.fcst_h5, out_h5))
            shutil.copy(self.fcst_h5, out_h5)

        with h5py.File(out_h5, 'a') as f_out:
            with Resource(self.actuals_h5, unscale=False) as f_in:
                logger.info('Correcting {} forecates'.format(self.fcst_dset))
                actuals = f_in[self.actuals_dset]

                ds = f_out[self.fcst_dset]
                fcst = ds[...]

                ds[...] = self._correct(fcst, actuals, fcst_perc=fcst_perc)

    @classmethod
    def correct(cls, fcst_h5, fcst_dset, out_h5,
                actuals_h5=None, actuals_dset=None, fcst_perc=None):
        """
        Bias correct and blend (if requested) forecasts using actuals

        Parameters
        ----------
        fcst_h5 : str
            Path to forecast .h5 file
        fcst_dset : str
            Dataset to correct
        out_h5 : str
            Output path for corrected .h5 file
        actuals_h5 : str, optional
            Path to forecast to .h5 file, by default None
        actuals_dset : str, optional
            Actuals dataset
        fcst_perc : float, None
            Percentage of forecast to use for blending, by default None
        """
        fcst = cls(fcst_h5, fcst_dset, actuals_h5=actuals_h5,
                   actuals_dset=actuals_dset)
        fcst.correct(out_h5, fcst_perc=fcst_perc)

    @classmethod
    def bias_correct(cls, fcst_h5, fcst_dset, out_h5,
                     actuals_h5=None, actuals_dset=None):
        """
        Bias correct forecast using actuals

        Parameters
        ----------
        fcst_h5 : str
            Path to forecast .h5 file
        fcst_dset : str
            Dataset to correct
        out_h5 : str
            Output path for corrected .h5 file
        actuals_h5 : str, optional
            Path to forecast to .h5 file, by default None
        actuals_dset : str, optional
            Actuals dataset
        """
        cls.correct(fcst_h5, fcst_dset, out_h5,
                    actuals_h5=actuals_h5, actuals_dset=actuals_dset)

    @classmethod
    def blend(cls, fcst_h5, fcst_dset, out_h5,
              actuals_h5=None, actuals_dset=None, fcst_perc=None):
        """
        Bias correct and blend forecast using actuals

        Parameters
        ----------
        fcst_h5 : str
            Path to forecast .h5 file
        fcst_dset : str
            Dataset to correct
        out_h5 : str
            Output path for corrected .h5 file
        actuals_h5 : str, optional
            Path to forecast to .h5 file, by default None
        actuals_dset : str, optional
            Actuals dataset
        fcst_perc : float
            Percentage of forecast to use for blending
        """
        cls.correct(fcst_h5, fcst_dset, out_h5,
                    actuals_h5=actuals_h5, actuals_dset=actuals_dset,
                    fcst_perc=fcst_perc)
