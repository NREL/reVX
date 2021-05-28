"""
Calculate costs to connect farms to existing grid. Process one or many SC
points.

5/21/2021
Mike Bannister
"""
from datetime import datetime as dt

import pandas as pd

from .path_finder import PathFinder
from .distance_calculators import SubstationDistanceCalculator, \
    TLineDistanceCalculator, LoadCenterDistanceCalculator, \
    SinkDistanceCalculator
from .config import SHORT_MULT, MEDIUM_MULT, SHORT_CUTOFF, MEDIUM_CUTOFF, \
    transformer_costs, NUM_LOAD_CENTERS, NUM_SINKS, iso_lookup
from .file_handlers import LoadData, FilterData
from .utilities import int_capacity


class DoThisForAllPowerClasses:
    # Theres only four, maybe do it by hand?
    pass


class DoThisForAllResolutions:
    # Theres only three, maybe do it by hand?
    pass


class ParallelProcessSCs:
    """
    Run a bunch, possibly with a slice
    """
    pass


class ProcessSCs:
    """
    Serial process tie line costs for Supply Curve points

    """
    def __init__(self, capacity_class='100MW', resolution=32, n=10):
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
        print(dt.now(), 'loading data')
        self._ld = LoadData(capacity_class, resolution)
        self._fd = FilterData(self._ld)

        rct = self._ld.rct
        subs_dc = SubstationDistanceCalculator(self._fd.subs, rct, n=n)
        tls_dc = TLineDistanceCalculator(self._fd.t_lines, rct, n=n)
        lcs_dc = LoadCenterDistanceCalculator(self._ld.lcs, rct,
                                              n=NUM_LOAD_CENTERS)
        sinks_dc = SinkDistanceCalculator(self._ld.sinks, rct, n=NUM_SINKS)

        self._cccfsc = CalcConnectCostsForSC(self._ld.costs_arr, subs_dc,
                                             tls_dc, lcs_dc, sinks_dc,
                                             capacity_class,
                                             self._ld.tie_voltage)
        print(dt.now(), 'done initing')

    def process(self, _slice=slice(None, None, None), plot=False):
        """
        Process all or a slice of SC points

        Parameters
        ----------
        _slice : Slice
            Slice of SC points to process. Defaults to all

        Returns
        -------
        all_costs : pandas.DataFrame
            Table of tie line costs to nearest transmission features for each
            SC point.
        """
        sc_points = self._ld.sc_points[_slice]

        all_costs = pd.DataFrame()
        for sc_pt in sc_points:
            print(dt.now(), f'--- Calculating costs for {sc_pt}')
            costs = self._cccfsc.calculate(sc_pt, plot=plot)
            print(dt.now(), f'Calculating costs complete')
            all_costs = pd.concat([all_costs, costs], axis=0)
        all_costs.reset_index(inplace=True, drop=True)
        return all_costs


class CalcConnectCostsForSC:
    """
    Calculate cost of connecting farm to existing grid. This includes line
    cost, all multipliers, new substations, substation upgrades and
    transformers. All transmission features should be valid for power class.
    """
    def __init__(self, costs_arr, subs_dc, tls_dc, lcs_dc, sinks_dc,
                 capacity_class, tie_voltage):
        """
        Parameters
        ----------
        costs_arr : numpy.ndarray
            Costs raster, value is cost in US$ to build line across cell,
            including all multipliers
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
        self._subs_dc = subs_dc
        self._tls_dc = tls_dc
        self._lcs_dc = lcs_dc
        self._sinks_dc = sinks_dc
        self._capacity_class = capacity_class
        self._tie_voltage = tie_voltage

        xfc = transformer_costs[str(tie_voltage)]
        self._xformer_costs = {int(k): v for k, v in xfc.items()}
        print(self._xformer_cost)

    def calculate(self, sc_pt, plot=False):
        """
        Calculate costs to connect supply curve point to existing transmission
        features

        Parameters
        ----------
        sc_pt : .file_handlers.SupplyCurvePoint
            Supply Curve point to calculate costs for

        Returns
        -------
        cdf : pandas.DataFrame
            Costs for tie lines, and new/upgraded substations include
            transformers to each existing grid feature.
        """
        pf = PathFinder.run(sc_pt, self._costs_arr, self._subs_dc,
                            self._tls_dc, self._lcs_dc, self._sinks_dc)
        cdf = pd.DataFrame([c.as_dict() for c in pf.costs])

        # Length multiplier
        cdf['length_mult'] = 1.0
        cdf.loc[cdf.length <= MEDIUM_CUTOFF, 'length_mult'] = MEDIUM_MULT
        cdf.loc[cdf.length < SHORT_CUTOFF, 'length_mult'] = SHORT_MULT
        cdf['adj_line_cost'] = cdf.tline_cost * cdf.length_mult

        # Transformer costs
        print(f'Processing transformer costs. v={self._tie_voltage}kV, '
              f'power={self._capacity_class}')
        cdf['xformer_cost_p_mw'] = cdf.apply(self._xformer_cost, axis=1)
        cdf['xformer_cost'] = cdf.xformer_cost_p_mw * \
            int_capacity(self._capacity_class)

        # Substation costs
        cdf['sub_upgrade_cost'] = cdf.apply(self._sub_upgrade_cost, axis=1)
        cdf['new_sub_cost'] = cdf.apply(self._new_sub_cost, axis=1)

        # Total cost
        cdf['total_cost'] = cdf.adj_line_cost + cdf.xformer_cost + \
            cdf.sub_upgrade_cost + cdf.new_sub_cost

        if plot:
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
            print(f'Failed to find proper transformer voltage for {row}, '
                  'defaulting to 500kV')
            v_class = 500

        cost_per_mw = self._xformer_costs[v_class]
        return cost_per_mw
