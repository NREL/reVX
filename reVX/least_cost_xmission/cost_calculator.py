"""
Calculate costs to connect farms to existing grid. Process one or many SC
points.

5/21/2021
Mike Bannister
"""
import logging
import math
from datetime import datetime as dt
from datetime import timedelta

import pandas as pd

from .path_finder import PathFinder
from .distance_calculators import SubstationDistanceCalculator, \
    TLineDistanceCalculator, LoadCenterDistanceCalculator, \
    SinkDistanceCalculator
from .config import SHORT_MULT, MEDIUM_MULT, SHORT_CUTOFF, MEDIUM_CUTOFF, \
    transformer_costs, NUM_LOAD_CENTERS, NUM_SINKS, iso_lookup, \
    new_sub_costs, upgrade_sub_costs, REPORTING_STEPS, min_power_classes, \
    SINK_CONNECTION_COST
from .file_handlers import LoadData, SupplyCurvePoint
from .utilities import int_capacity

logger = logging.getLogger(__name__)


class ProcessSCs:
    """
    Serial process tie line costs for Supply Curve points

    """
    def __init__(self, capacity_class='100MW', resolution=128, n=10):
        """
        Parameters
        ----------
        capacity_class : String
            Desired reV power capacity class, one of "100MW", "200MW", "400MW",
            "1000MW"
        resolution : Int
            Desired Supply Curve Point resolution, one of: 32, 64, 128
        n : Int
            Number of nearby transmission features to search for
        """
        self._capacity_class = capacity_class
        logger.info('Loading data')
        self.ld = LoadData(capacity_class, resolution=resolution)

        rct = self.ld.rct
        subs_dc = SubstationDistanceCalculator(self.ld.subs, rct, n=n)
        tls_dc = TLineDistanceCalculator(self.ld.t_lines, rct, n=n)
        lcs_dc = LoadCenterDistanceCalculator(self.ld.lcs, rct,
                                              n=NUM_LOAD_CENTERS)
        sinks_dc = SinkDistanceCalculator(self.ld.sinks, rct, n=NUM_SINKS)

        self._cccfsc = CalcConnectCostsForSC(self.ld.costs_arr,
                                             self.ld.paths_arr, subs_dc,
                                             tls_dc, lcs_dc, sinks_dc,
                                             capacity_class,
                                             self.ld.tie_voltage)
        logger.info('Finished loading data')

    def process(self, sc_pts=None, plot=False, chunk_id=''):
        """
        Process all or a subset of SC points

        Parameters
        ----------
        sc_pts : List | Slice | None
            List of SC points to process. List may be a list of SC point
            ids (ints) or SupplyCurvePoint instances. If Slice, process
            that slice of the SC points. Process all points if None
        plot : bool
            Plot graphs if true
        chunk_id : str
            Id of chunk being run for multiprocessing logging

        Returns
        -------
        all_costs : pandas.DataFrame
            Table of tie line costs to nearest transmission features for each
            SC point.
        """
        if isinstance(sc_pts, list) and isinstance(sc_pts[0], int):
            sc_pts = [x for x in self.ld.sc_points if x.id in sc_pts]
        elif isinstance(sc_pts, slice):
            sc_pts = self.ld.sc_points[sc_pts]
        elif sc_pts is None:
            sc_pts = self.ld.sc_points
        elif isinstance(sc_pts, list) and \
             isinstance(sc_pts[0], SupplyCurvePoint):
            pass
        else:
            msg = 'sc_pts must be either list, slice, or None. ' +\
                  f'Got {type(sc_pts)}'
            logger.error(msg)
            raise AttributeError(msg)

        # Keep track of run times and report progress
        step = math.ceil(len(sc_pts)/REPORTING_STEPS)
        report_steps = range(step-1, len(sc_pts), step)
        run_times = []

        plot_costs_arr = None
        if plot:
            plot_costs_arr = self.ld.plot_costs_arr

        all_costs = pd.DataFrame()
        for i, sc_pt in enumerate(sc_pts):
            now = dt.now()
            costs = self._cccfsc.calculate(sc_pt, plot_costs_arr)
            all_costs = pd.concat([all_costs, costs], axis=0)
            run_times.append(dt.now() - now)

            if i in report_steps:
                progress = int((i+1)/len(sc_pts)*100)
                avg = sum(run_times, timedelta(0))/len(run_times)
                left = (len(sc_pts)-i-1)*avg
                msg = (f'{chunk_id}Finished SC pt {sc_pt.id} ({i+1} of '
                       f'{len(sc_pts)}). {progress}% '
                       f'complete. Average time of {avg} per SC pt. '
                       f'Approx {left} left for this chunk.')
                logger.info(msg)
        all_costs.reset_index(inplace=True, drop=True)

        avg = sum(run_times, timedelta(0))/len(run_times)
        msg = (f'{chunk_id}Finished processing chunk ({i+1} of '
               f'{len(sc_pts)} pts). Average time of {avg} per SC pt.')
        logger.info(msg)

        all_costs['max_cap'] = int_capacity(self._capacity_class)
        all_costs['min_cap'] = min_power_classes[self._capacity_class]
        return all_costs


class CalcConnectCostsForSC:
    """
    Calculate cost of connecting farm to existing grid. This includes line
    cost, all multipliers, new substations, substation upgrades and
    transformers. All transmission features should be valid for power class.
    """
    def __init__(self, costs_arr, paths_arr, subs_dc, tls_dc, lcs_dc,
                 sinks_dc, capacity_class, tie_voltage):
        """
        Parameters
        ----------
        costs_arr : numpy.ndarray
            Costs raster, value is cost in US$ to build line across cell,
            including all multipliers
        paths_arr : numpy.ndarray
            Costs raster include transmission barriers multiplier
        subs_dc : SubstationDistanceCalculator
            Distance calculator for substations
        tls_dc : TlineDistanceCalculator
            Distance calculator for existing transmission lines
        lcs_dc : LoadCenterDistanceCalculator
            Distance calculator for load centers
        sinks_dc : SinkDistanceCalculator
            Distance calculator for sinks
        capacity_class : String
            Desired reV power capacity class, one of "100MW", "200MW", "400MW",
            "1000MW"
        tie_voltage : int
            Line voltage for capacity_class (kV)
        """
        self._costs_arr = costs_arr
        self._paths_arr = paths_arr
        self._subs_dc = subs_dc
        self._tls_dc = tls_dc
        self._lcs_dc = lcs_dc
        self._sinks_dc = sinks_dc
        self._capacity_class = capacity_class
        self._tie_voltage = tie_voltage

        xfc = transformer_costs[str(tie_voltage)]
        self._xformer_costs = {int(k): v for k, v in xfc.items()}
        self._reverse_iso = {v: k for k, v in iso_lookup.items()}

    def calculate(self, sc_pt, plot_costs_arr=None):
        """
        Calculate costs to connect supply curve point to existing transmission
        features

        Parameters
        ----------
        sc_pt : .file_handlers.SupplyCurvePoint
            Supply Curve point to calculate costs for
        plot_costs_arr : numpy.ndarray | None
            Costs raster with transmission barriers layer included for plotting

        Returns
        -------
        cdf : pandas.DataFrame
            Costs for tie lines, and new/upgraded substations include
            transformers to each existing grid feature.
        """

        pf = PathFinder.run(sc_pt, self._costs_arr, self._paths_arr,
                            self._subs_dc, self._tls_dc, self._lcs_dc,
                            self._sinks_dc, plot_costs_arr=plot_costs_arr)
        cdf = pf.costs

        # Length multiplier
        cdf['length_mult'] = 1.0
        cdf.loc[cdf.dist_km <= MEDIUM_CUTOFF, 'length_mult'] = MEDIUM_MULT
        cdf.loc[cdf.dist_km < SHORT_CUTOFF, 'length_mult'] = SHORT_MULT
        cdf['tie_line_cost'] = cdf.raw_line_cost * cdf.length_mult

        # Transformer costs
        cdf['xformer_cost_p_mw'] = cdf.apply(self._xformer_cost, axis=1)
        cdf['xformer_cost'] = cdf.xformer_cost_p_mw * \
            int_capacity(self._capacity_class)

        # Substation costs
        cdf['sub_upgrade_cost'] = cdf.apply(self._sub_upgrade_cost, axis=1)
        cdf['new_sub_cost'] = cdf.apply(self._new_sub_cost, axis=1)

        # Sink costs
        cdf.loc[cdf.category == 'PCALoadCen', 'new_sub_cost'] = \
            SINK_CONNECTION_COST

        # Total cost
        cdf['connection_cost'] = cdf.xformer_cost + cdf.sub_upgrade_cost +\
            cdf.new_sub_cost
        cdf['trans_cap_cost'] = cdf.tie_line_cost + cdf.connection_cost

        if plot_costs_arr is not None:
            pf.plot_paths()

        return cdf

    def _sub_upgrade_cost(self, row):
        """
        Calculate upgraded substation cost for substations and load centers

        Parameters
        ----------
        row : pandas.DataFrame row
            Cost row for one tie-line

        Returns
        -------
        cost : float
            Cost to upgrade substation
        """
        if row.category == 'Substation' or row.category == 'LoadCen':
            volts = str(self._tie_voltage)
            return upgrade_sub_costs[self._reverse_iso[row.region]][volts]

        return 0

    def _new_sub_cost(self, row):
        """
        Calculate cost to build new substation

        Parameters
        ----------
        row : pandas.DataFrame row
            Cost row for one tie-line

        Returns
        -------
        cost : float
            Cost to build new substation
        """
        if row.category == 'TransLine':
            volts = str(self._tie_voltage)
            return new_sub_costs[self._reverse_iso[row.region]][volts]

        return 0

    def _xformer_cost(self, row):
        """
        Calculate transformer cost

        Parameters
        ----------
        row : pandas.DataFrame row
            Cost row for one tie-line

        Returns
        -------
        cost : float
            Cost of transformer to bring tie line up to existing trans volts
        """
        if self._tie_voltage >= row.min_volts:
            return 0

        if row.region == iso_lookup['TEPPC']:
            return 0

        # If min_voltage is not in lookup table, get next largest value
        v_class = 0
        for volts in self._xformer_costs.keys():
            if volts <= row.min_volts:
                v_class = volts
            else:
                break

        if v_class == 0:
            logger.warning('Failed to find proper transformer voltage for '
                           f'{row}, defaulting to 500kV')
            v_class = 500

        cost_per_mw = self._xformer_costs[v_class]
        return cost_per_mw
