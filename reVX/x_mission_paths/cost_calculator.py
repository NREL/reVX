"""
Calculate costs to connect farms to existing grid. Process one or many SC
points.

5/21/2021
Mike Bannister
"""
from datetime import datetime as dt

import pandas as pd

from .path_finder import SubstationDistanceCalculator, \
    TLineDistanceCalculator, PathFinder
from .config import SHORT_MULT, MEDIUM_MULT, SHORT_CUTOFF, MEDIUM_CUTOFF
from .file_handlers import LoadData, FilterData


class DoThisForAllPowerClasses:
    # Theres only four, maybe do it by hand?
    pass


class DoThisForAllResolutions:
    # Theres only four, maybe do it by hand?
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
    def __init__(self, capacity, resolution=32, n=10):
        """
        Parameters
        ----------
        capacity : String
            Desired reV power capacity class, one of "100MW", "200MW", "400MW",
            "1000MW"
        resolution : Int
            Desired Supply Curve Point resolution, one of: 32, 64, 128
        n : Int
            Number of nearby transmission features to search for
        """
        print(dt.now(), 'loading data')
        self._ld = LoadData(capacity, resolution)
        print(dt.now(), 'filtering data')
        self._fd = FilterData(self._ld)

        print(dt.now(), 'creating distance calculators')
        rct = self._ld.rct
        subs_dc = SubstationDistanceCalculator(self._fd.subs, rct, n=n)
        tls_dc = TLineDistanceCalculator(self._fd.t_lines, rct, n=n)
        self._cccfsc = CalcConnectCostsForSC(self._ld.costs_arr, subs_dc,
                                             tls_dc)
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
            print(dt.now(), f'Calculating costs for {sc_pt}')
            costs = self._cccfsc.calculate(sc_pt, plot=plot)
            print(dt.now(), f'Calculating costs complete')
            all_costs = pd.concat([all_costs, costs], axis=0)
        all_costs.reset_index(inplace=True)
        return all_costs


class CalcConnectCostsForSC:
    """
    Calculate cost of connecting farm to existing grid. This includes line
    cost, all multipliers, new substations, substation upgrades and
    transformers. All transmission features should be valid for power class.
    """
    def __init__(self, costs_arr, subs_dc, tls_dc):
        """
        Parameters
        ----------
        costs_arr : numpy.ndarray
            Costs raster, value is cost in US$ to build line across cell,
            including all multipliers
        subs_dc : .path_finder.SubstationDistanceCalculator
            Distance calculator for substations
        tls_dc : .path_finder.TlineDistanceCalculator
            Distance calculator for existing transmission lines
        """
        self._costs_arr = costs_arr
        self._subs_dc = subs_dc
        self._tls_dc = tls_dc

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
                            self._tls_dc)
        costs = pf.costs
        cdf = pd.DataFrame([tc.as_dict() for tc in costs])

        # Length multiplier
        cdf['length_mult'] = 1.0
        cdf.loc[cdf.length <= MEDIUM_CUTOFF, 'length_mult'] = MEDIUM_MULT
        cdf.loc[cdf.length < SHORT_CUTOFF, 'length_mult'] = SHORT_MULT
        cdf['adj_line_cost'] = cdf.tline_cost * cdf.length_mult

        # TODO - add transformer and substation costs

        if plot:
            pf.plot_paths()

        return cdf
