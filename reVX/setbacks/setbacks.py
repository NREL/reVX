# -*- coding: utf-8 -*-
"""
Setback exclusion computation classes
"""
import logging

from reVX.setbacks.base import AbstractBaseSetbacks, BUFFERS, FEATURE_FILTERS


logger = logging.getLogger(__name__)


def _format_str(in_str):
    """Format a string like regulation class does. """
    return in_str.strip().lower().replace("-", " ").replace("_", " ")


def _camel_case_str(in_str):
    """Format a string into CamelCase"""
    return _format_str(in_str).title().replace(" ", "")


def setbacks_calculator(feature_type, buffer_type="default",
                        feature_filter_type="centroid",
                        feature_subtypes_to_exclude=None,
                        num_features_per_worker=10_000):
    """Generate a setback calculator for a particular feature.

    Parameters
    ----------
    feature_type : str | list
        String or list of strings that represent how this feature is
        distinguished in the local regulations file. In particular,
        this string (or list of strings) should match the values under
        the ``Feature Type`` column of the setback regulations file
        corresponding to this feature. For example, specifying
        ``feature_type="property line"`` would mean that this setback
        calculator looks for ``"property line"`` regulations (i.e.
        parcel setbacks) in the input regulations file. This input is
        case insensitive (i.e. "ROADS", "Roads", "roads", and "rOAds"
        are all equivalent), and spaces, dashes, and underscores are
        interchangeable (i.e. "property line", "property_line", and
        "property-line" are all equivalent).
    buffer_type : str, optional
        Name of the buffer type to use for this setback calculation. See
        :attr:`~reVX.setbacks.base.BUFFERS` for all available options -
        this input must be a key from that dictionary. In most cases,
        the default positive buffer is suitable.
        By default, ``"default"``.
    feature_filter_type : str, optional
        Name of the feature filter type to use for this setback
        calculation. Typically, features like structures or parcels are
        filtered using "centroid", which only applies setback
        regulations to features whose centroid is within the county with
        the ordinance. On the other hand, features such as roads,
        railroads transmission lines, etc., are filtered using "clip",
        which clips the feature to the county boundaries before applying
        a setback. This value must be a key in the
        :attr:`~reVX.setbacks.base.FEATURE_FILTERS` dictionary.
        By default, ``"centroid"``.
    feature_subtypes_to_exclude : list | set, optional
        A list of strings specifying the feature subtypes that should be
        excluded from the local regulations calculation.
        By default, ``None``.
    num_features_per_worker : int, optional
        Number of features each worker (CPU processor) on a node
        processes. By default, ``10,000``.

    Returns
    -------
    class
        A class object that can be used to instantiate the setbacks
        calculator. The initializer parameters are identical to that of
        :class:`~reVX.setbacks.base.AbstractBaseSetbacks`.
    """
    if isinstance(feature_type, str):
        feature_type = [feature_type]

    camel_case_feature = "".join(map(_camel_case_str, sorted(feature_type)))
    feature_type = set(map(_format_str, feature_type))

    if buffer_type not in BUFFERS:
        msg = ("Unknown buffer type specified: {!r}. Must be one of {}"
               .format(buffer_type, list(BUFFERS)))
        logger.error(msg)
        raise ValueError(msg)

    if feature_filter_type not in FEATURE_FILTERS:
        msg = ("Unknown feature filter type specified: {!r}. Must be one of {}"
               .format(feature_filter_type, list(FEATURE_FILTERS)))
        logger.error(msg)
        raise ValueError(msg)

    feature_subtypes_to_exclude = set(feature_subtypes_to_exclude or set())
    num_features_per_worker = int(max(0, num_features_per_worker) or 10_000)
    class_name = "{}Setbacks".format(camel_case_feature)
    class_attrs = {"FEATURE_TYPES": feature_type,
                   "FEATURE_SUBTYPES_TO_EXCLUDE": feature_subtypes_to_exclude,
                   "BUFFER_TYPE": buffer_type,
                   "FEATURE_FILTER_TYPE": feature_filter_type,
                   "NUM_FEATURES_PER_WORKER": num_features_per_worker}

    return type(class_name, (AbstractBaseSetbacks,), class_attrs)


SETBACK_SPECS = {
    "parcel": {
        "feature_type": "property line",
        "buffer_type": "parcel",
        "feature_filter_type": "centroid",
        "feature_subtypes_to_exclude": None,
        "num_features_per_worker": 10_000,
    },
    "rail": {
        "feature_type": "railroads",
        "buffer_type": "default",
        "feature_filter_type": "clip",
        "feature_subtypes_to_exclude": None,
        "num_features_per_worker": 10_000,
    },
    "road": {
        "feature_type": ['roads', 'highways', 'highways 111'],
        "buffer_type": "default",
        "feature_filter_type": "clip",
        "feature_subtypes_to_exclude": None,
        "num_features_per_worker": 10_000,
    },
    "structure": {
        "feature_type": "structures",
        "buffer_type": "default",
        "feature_filter_type": "centroid",
        "feature_subtypes_to_exclude": ["Occupied Community Buildings"],
        "num_features_per_worker": 10_000,
    },
    "transmission": {
        "feature_type": "transmission",
        "buffer_type": "default",
        "feature_filter_type": "clip",
        "feature_subtypes_to_exclude": None,
        "num_features_per_worker": 10_000,
    },
    "water": {
        "feature_type": "water",
        "buffer_type": "default",
        "feature_filter_type": "clip",
        "feature_subtypes_to_exclude": None,
        "num_features_per_worker": 700,
    },
}
"""Default Setback Calculator Specifications.

These calculators are available by default. To use them, simply match
the key in this dictionary with your dataset path.
"""


SETBACKS = {key: setbacks_calculator(**kwargs)
            for key, kwargs in SETBACK_SPECS.items()}
