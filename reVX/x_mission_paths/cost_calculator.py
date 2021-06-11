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
    new_sub_costs, upgrade_sub_costs, REPORTING_STEPS
from .file_handlers import LoadData
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

    def process(self, indices=None, plot=False, chunk_id=''):
        """
        Process all or a subset of SC points

        Parameters
        ----------
        indices : List | None
            List of SC point indices to process. Process all if None
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
        sc_points = self.ld.sc_points

        if indices is None:
            indices = range(len(sc_points))

        # keep track of run times and report progress
        step = math.ceil(len(indices)/REPORTING_STEPS)
        report_steps = indices[::step][1:]
        run_times = []

        plot_costs_arr = None
        if plot:
            plot_costs_arr = self.ld.plot_costs_arr

        all_costs = pd.DataFrame()
        for i, index in enumerate(indices):
            now = dt.now()
            sc_pt = sc_points[index]
            costs = self._cccfsc.calculate(sc_pt, plot_costs_arr)
            all_costs = pd.concat([all_costs, costs], axis=0)
            run_times.append(dt.now() - now)

            if index in report_steps:
                progress = int(i/len(indices)*100)
                avg = sum(run_times, timedelta(0))/len(run_times)
                left = (len(indices)-i)*avg
                msg = (f'{chunk_id}Finished SC pt {sc_pt.id} ({i} of '
                       f'{len(indices)}). {progress}% '
                       f'complete. Average time of {avg} per SC pt. '
                       f'Approx {left} left for this chunk.')
                logger.info(msg)
        all_costs.reset_index(inplace=True, drop=True)
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
        cdf = pd.DataFrame([c.as_dict() for c in pf.costs])

        # Length multiplier
        cdf['length_mult'] = 1.0
        cdf.loc[cdf.length <= MEDIUM_CUTOFF, 'length_mult'] = MEDIUM_MULT
        cdf.loc[cdf.length < SHORT_CUTOFF, 'length_mult'] = SHORT_MULT
        cdf['adj_line_cost'] = cdf.tline_cost * cdf.length_mult

        # Transformer costs
        cdf['xformer_cost_p_mw'] = cdf.apply(self._xformer_cost, axis=1)
        cdf['xformer_cost'] = cdf.xformer_cost_p_mw * \
            int_capacity(self._capacity_class)
        # TODO int_capacity might not be needed

        # Substation costs
        cdf['sub_upgrade_cost'] = cdf.apply(self._sub_upgrade_cost, axis=1)
        cdf['new_sub_cost'] = cdf.apply(self._new_sub_cost, axis=1)

        # Total cost
        cdf['total_cost'] = cdf.adj_line_cost + cdf.xformer_cost + \
            cdf.sub_upgrade_cost + cdf.new_sub_cost

        if plot_costs_arr is not None:
            pf.plot_paths()

        return cdf

    def _sub_upgrade_cost(self, row):
        """
        Calculate upgraded substation cost

        Parameters
        ----------
        row : pandas.DataFrame row
            Cost row for one tie-line

        Returns
        -------
        cost : float
            Cost to upgrade substation
        """
        if row.trans_type == 'sub':
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
        if row.trans_type == 't-line':
            volts = str(self._tie_voltage)
            return new_sub_costs[self._reverse_iso[row.region]][volts]

        return 0

    def _xformer_cost(self, row):
        """
        Calculate transformer cost

        Parameters
        ----------
        TODO

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
