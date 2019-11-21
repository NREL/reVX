# -*- coding: utf-8 -*-
"""
Clustering Methods
"""
from sklearn.cluster import KMeans


class ClusteringMethods:
    """ Base class of clustering methods """

    @staticmethod
    def kmeans(data, **kwargs):
        """ Cluster based on kmeans methodology """

        kmeans = KMeans(random_state=0, **kwargs)
        results = kmeans.fit(data)
        return results.labels_
