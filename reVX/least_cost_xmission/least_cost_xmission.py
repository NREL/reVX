# -*- coding: utf-8 -*-
"""
Module to compute least cost xmission paths, distances, and costs
"""


class TieLineCosts:
    """
    Compute least cost tie-line path to all features to be connected a single
    supply curve point
    """
    def __init__(self, least_cost_array, cost_array, trans_features):
        """
        Parameters
        ----------
        least_cost_array : ndarray
            Clipped least cost array for MCPGeometric
        cost_array : ndarray
            Clipped raw cost array
        trans_features : pandas.DataFrame
            DataFrame of transmission features to connect to supply curve point
            Includes row, col indices of features relative to the clipped
            cost arrays
        """


class LeastCostXmission:
    """
    Compute Least Cost tie-line paths and full transmission cap cost
    for all possible connections to all supply curve points
    -
    """
    def __init__(self, cost_fpath, features_fpath, resolution=128,
                 dist_thresh=None):
        """
        - Load trans features from shape file
        - Map all features (except lines) to row, col indices of cost
        domain/raster
        - compute deterministic sc_point based on resolution
        - Reduce sc_points based on distance threshold + resolution * 90m and
        'dist_to_coast' layer
        - For each sc_point determine distance to 2 nearest PCA load centers
        (sinks), use as clipping distance
        - Clip raster and reduce tranmission table, pass to TieLineCosts class
        to compute path/costs
        - Combine tables for all sc_points
        - Compute connections costs
        - Dump to .csv

        Parameters
        ----------
        cost_fpath : [type]
            [description]
        features_fpath : [type]
            [description]
        resolution : int, optional
            [description], by default 128
        dist_thresh : [type], optional
            [description], by default None
        """
