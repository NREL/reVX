# -*- coding: utf-8 -*-
"""
Tie Line Costs tests
"""
import logging
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
import os
import pytest

from reVX.least_cost_xmission.config import (TRANS_LINE_CAT, LOAD_CENTER_CAT,
                                             SINK_CAT, SUBSTATION_CAT)
# from reVX.least_cost_xmission.trans_cap_costs import (TieLineCosts,
#                                                       TransCapCosts)
# from reVX import TESTDATADIR

logger = logging.getLogger(__name__)

# COST_H5 = os.path.join(TESTDATADIR, 'xmission', 'xmission_layers.h5')
# FEATURES = os.path.join(TESTDATADIR, 'xmission', 'ri_allconns.gpkg')


def plot_paths(trans_cap_costs, x_feats, cmap='viridis', label=False,
               plot_paths_arr=True):
    """
    Plot least cost paths for QAQC
    """
    plt.figure(figsize=(30, 15))
    if plot_paths_arr:
        trans_cap_costs._mcp_cost[trans_cap_costs._mcp_cost == np.inf] = 0.1
        norm = colors.LogNorm(vmin=trans_cap_costs._mcp_cost.min(),
                              vmax=trans_cap_costs._mcp_cost.max())
        plt.imshow(trans_cap_costs._mcp_cost, cmap=cmap, norm=norm)
    else:
        plt.imshow(trans_cap_costs._cost, cmap=cmap)

    plt.colorbar()

    # Plot paths
    for _, feat in x_feats.iterrows():

        name = feat.category[0] + str(feat.trans_gid)
        try:
            indices = trans_cap_costs._mcp.traceback((feat.row, feat.col))
        except ValueError:
            # No path to trans feature.
            name = feat.category[0] + str(feat.trans_gid)
            msg = ("Can't find path to {} {} from "
                   "SC pt {}".format(feat.category, feat.trans_gid,
                                     trans_cap_costs._sc_point.name))
            logger.error(msg)
            continue

        path_xs = [x[1] for x in indices]
        path_ys = [x[0] for x in indices]
        plt.plot(path_xs, path_ys, color='white')

    # Plot trans features
    style = {
        SUBSTATION_CAT: {
            'marker': 'd',
            'color': 'red',
            't_offset': 0,
        },
        TRANS_LINE_CAT: {
            'marker': '^',
            'color': 'lightblue',
            't_offset': 50,
        },
        LOAD_CENTER_CAT: {
            'marker': 'v',
            'color': 'green',
            't_offset': 0,
        },
        SINK_CAT: {
            'marker': 'X',
            'color': 'orange',
            't_offset': 0,
        },
    }

    path_effects = [PathEffects.withStroke(linewidth=3, foreground='w')]

    for _, feat in x_feats.iterrows():
        marker = style[feat.category]['marker']
        color = style[feat.category]['color']
        offset = style[feat.category]['t_offset']
        name = feat.category[0] + str(feat.trans_gid)

        if label:
            plt.text(feat.col + 20, feat.row + offset, name, color='black',
                     path_effects=path_effects, fontdict={'size': 13})
        plt.plot(feat.col, feat.row, marker=marker, color=color)

    # Plot sc_point
    plt.plot(trans_cap_costs.start[1], trans_cap_costs.start[0], marker='P',
             color='black', markersize=18)
    plt.plot(trans_cap_costs.start[1], trans_cap_costs.start[0], marker='P',
             color='yellow', markersize=10)

    plt.title(str(trans_cap_costs._sc_point.name))
    plt.show()


def execute_pytest(capture='all', flags='-rapP'):
    """Execute module as pytest with detailed summary report.

    Parameters
    ----------
    capture : str
        Log or stdout/stderr capture option. ex: log (only logger),
        all (includes stdout/stderr)
    flags : str
        Which tests to show logs and results for.
    """

    fname = os.path.basename(__file__)
    pytest.main(['-q', '--show-capture={}'.format(capture), fname, flags])


if __name__ == '__main__':
    execute_pytest()
