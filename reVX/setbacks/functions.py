# -*- coding: utf-8 -*-
"""
Helper functions for setback exclusion computation
"""

import geopandas as gpd


def positive_buffer(features, regulation_value):
    """Buffer features using a given regulation value.

    This function applies a simple positive buffer to every polygon in
    the input features.

    Parameters
    ----------
    features : geopandas.GeoDataFrame
        Features to apply buffer to.
    regulation_value : int | float
        Regulations value used to buffer the features. This value should
        be in the same units as the ``features`` input GeoDataFrame.

    See Also
    --------
    geopandas.GeoSeries.buffer : Function used ot buffer each feature.

    Returns
    -------
    list
        List of buffered buffered feature shapes.
    """
    return list(features.buffer(regulation_value))


def parcel_buffer(features, regulation_value):
    """Buffer features imitating a parcel setback.

    Parcel (property-line) setbacks typically prohibit any type of build
    within "x" meters of a property line. Therefore, this type of buffer
    first applies a negative buffer on each input feature and then takes
    the difference between the original and the negatively-buffered
    feature. The resulting shape is the "inside" of the original input
    feature, where each edge is no closer than ``regulation_value`` to
    the original feature boundary.

    Parameters
    ----------
    features : geopandas.GeoDataFrame
        Features to apply buffer to.
    regulation_value : int | float
        Regulations value used to (negatively) buffer the features.
        This value should be in the same units as the ``features`` input
        GeoDataFrame.

    See Also
    --------
    geopandas.GeoSeries.buffer : Function used ot buffer each feature.

    Returns
    -------
    list
        List of buffered buffered feature shapes.
    """
    negative_buffer = features.buffer(-1 * regulation_value)
    return list(features.buffer(0).difference(negative_buffer))


def features_with_centroid_in_county(features, county):
    """Filter features to those with centroids within the given county.

    Parameters
    ----------
    features : geopandas.GeoDataFrame
        Features to setback from.
    county : geopandas.GeoDataFrame
        Regulations for a single county.

    Returns
    -------
    features : geopandas.GeoDataFrame
        Features that have centroid in county.
    """
    mask = features.centroid.within(county['geometry'].values[0])
    return features.loc[mask]


def features_clipped_to_county(features, county):
    """Clip features to the given county geometry.

    Parameters
    ----------
    features : geopandas.GeoDataFrame
        Features to setback from.
    county : geopandas.GeoDataFrame
        Regulations for a single county.

    Returns
    -------
    features : geopandas.GeoDataFrame
        Features clipped to county geometry.
    """
    tmp = gpd.clip(features, county)
    return tmp[~tmp.is_empty]
