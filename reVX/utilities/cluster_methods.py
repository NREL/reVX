# -*- coding: utf-8 -*-
"""
Clustering Methods
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


class ClusteringMethods:
    """ Base class of clustering methods """

    @staticmethod
    def _normalize_values(arr, norm=None, axis=None):
        """
        Normalize values in array by column

        Parameters
        ----------
        arr : ndarray
            ndarray of values extracted from meta
            shape (n samples, with m features)
        norm : str
            Normalization method to use (see sklearn.preprocessing.normalize)
            if None range normalize
        axis : int
            Axis to normalize along

        Returns
        ---------
        arr : ndarray
            array with values normalized by column
            shape (n samples, with m features)
        """
        if norm:
            arr = normalize(arr, norm=norm, axis=axis)
        else:
            if np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(float)

            min_all = arr.min(axis=axis)
            max_all = arr.max(axis=axis)
            range_all = max_all - min_all
            if axis is not None:
                pos = range_all == 0
                range_all[pos] = 1

            arr -= min_all
            arr /= range_all

        return arr

    @staticmethod
    def kmeans(data, **kwargs):
        """ Cluster based on kmeans methodology """

        kmeans = KMeans(random_state=0, **kwargs)
        results = kmeans.fit(data)
        labels = results.labels_

        # Create deterministic cluster labels based on size
        label_n, l_size = np.unique(labels, return_counts=True)
        idx = np.argsort(l_size)
        l_mapping = dict(zip(label_n[idx], label_n))

        sorted_labels = labels.copy()
        for k, v in l_mapping.items():
            sorted_labels[labels == k] = v

        return sorted_labels
