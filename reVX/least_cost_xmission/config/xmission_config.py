"""
Load default configuration for tie-line cost determination

Mike Bannister
7/27/21
"""
import logging
import os
import numpy as np

from rex.utilities.utilities import safe_json_load

CONFIG_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG = {
    "base_line_costs": os.path.join(CONFIG_DIR, 'base_line_costs.json'),
    "iso_lookup": os.path.join(CONFIG_DIR, 'iso_lookup.json'),
    "iso_multipliers": os.path.join(CONFIG_DIR, 'iso_multipliers.json'),
    "land_use_classes": os.path.join(CONFIG_DIR, 'land_use_classes.json'),
    "new_substation_costs": os.path.join(CONFIG_DIR,
                                         'new_substation_costs.json'),

    "power_classes": os.path.join(CONFIG_DIR, 'power_classes.json'),
    "power_to_voltage": os.path.join(CONFIG_DIR, 'power_to_voltage.json'),
    "transformer_costs": os.path.join(CONFIG_DIR,
                                      'transformer_costs.json'),
    "upgrade_substation_costs": os.path.join(CONFIG_DIR,
                                             'upgrade_substation_costs.json')
}
logger = logging.getLogger(__name__)


class XmissionConfig(dict):
    """ Load default tie lines configuration """

    def __init__(self, config=None):
        """
        Use default values for all parameters if None.

        Parameters
        ----------
        config : str | dict, optional
            Path to json file containing Xmission cost configuration values,
            or jsonified dict of cost configuration values,
            or dictionary of configuration values,
            or dictionary of paths to config jsons,
            if None use defaults in './config.json'
        """
        super().__init__()

        self._load_config(CONFIG)
        self._load_config(config)

    def __getitem__(self, k):
        if k == 'reverse_iso':
            out = {v: k for k, v in self['iso_lookup'].items()}
        elif k == 'voltage_to_power':
            out = {v: k for k, v in self['power_to_voltage'].items()}
        elif k == 'line_power_to_classes':
            out = {v: k for k, v in self['power_classes'].items()}
        else:
            out = super().__getitem__(k)

        return out

    @staticmethod
    def _parse_cap_class(capacity):
        """
        Parse capacity class from input capacity which can be a number or a
        string

        Parameters
        ----------
        capacity : int | float | str
            Capacity of interest

        Returns
        -------
        cap_class : str
            Capacity class in format "{capacity}MW"
        """
        if not isinstance(capacity, str):
            cap_class = '{}MW'.format(int(capacity))
        elif not capacity.endswith('MW'):
            cap_class = capacity + 'MW'
        else:
            cap_class = capacity

        return cap_class

    def _load_config(self, config):
        """
        Load config

        Parameters
        ----------
        config : str | dict
            Path to json file containing Xmission cost configuration values,
            or jsonified dict of cost configuration values,
            or dictionary of configuration values,
            or dictionary of paths to config jsons
        """
        if config is not None:
            if isinstance(config, str):
                config = safe_json_load(config)

            if not isinstance(config, dict):
                msg = ('Xmission costs config must be a path to a json file, '
                       'a jsonified dictionary, or a dictionary, not: {}'
                       .format(config))
                logger.error(msg)
                raise ValueError(msg)

            for k, v in config.items():
                if v.endswith('.json'):
                    v = safe_json_load(v)

                self[k] = v

    def capacity_to_kv(self, capacity):
        """
        Convert capacity class to line voltage

        Parameters
        ----------
        capacity : int
            Capacity class in MW

        Returns
        -------
        kv : int
            Tie-line voltage in kv
        """
        cap_class = self._parse_cap_class(capacity)
        line_capacity = self['power_classes'][cap_class]
        kv = self['power_to_voltage'][str(line_capacity)]

        return int(kv)

    def kv_to_capacity(self, kv):
        """
        Convert line voltage to capacity class

        Parameters
        ----------
        kv : in
            Tie-line voltage in kv

        Returns
        -------
        capacity : int
            Capacity class in MW
        """
        line_capacity = self['voltage_to_power'][kv]
        capacity = self['line_power_to_classes'][line_capacity].strip("MW")

        return int(capacity)

    def sub_upgrade_cost(self, region, tie_line_voltage):
        """
        Extract substation upgrade costs in $ for given region and tie-line
        voltage rating

        Parameters
        ----------
        region : int
            Region code, used to extract ISO
        tie_line_voltage : int | str
            Tie-line voltage class in kV

        Returns
        -------
        int
            Substation upgrade cost
        """
        if not isinstance(tie_line_voltage, str):
            tie_line_voltage = str(tie_line_voltage)

        region = self['reverse_iso'][region]

        return self['upgrade_substation_costs'][region][tie_line_voltage]

    def new_sub_cost(self, region, tie_line_voltage):
        """
        Extract new substation costs in $ for given region and tie-line
        voltage rating

        Parameters
        ----------
        region : int
            Region code, used to extract ISO
        tie_line_voltage : int | str
            Tie-line voltage class in kV

        Returns
        -------
        int
            New substation cost
        """
        if not isinstance(tie_line_voltage, str):
            tie_line_voltage = str(tie_line_voltage)

        region = self['reverse_iso'][region]

        return self['new_substation_costs'][region][tie_line_voltage]

    def xformer_cost(self, feature_voltage, tie_line_voltage):
        """
        Extract transformer costs in $ for given region and tie-line
        voltage rating

        Parameters
        ----------
        feature_voltage : int
            Voltage of feature that tie-line is connecting to
        tie_line_voltage : int | str
            Tie-line voltage class in kV

        Returns
        -------
        int
            Transformer cost as $/MW
        """
        if not isinstance(tie_line_voltage, str):
            tie_line_voltage = str(tie_line_voltage)

        costs = self['transformer_costs'][tie_line_voltage]

        classes = np.array(sorted(map(int, costs)))
        valid_idx = np.where(classes >= feature_voltage)[0]
        if valid_idx.size:
            v_class = classes[valid_idx[0]]
        else:
            v_class = classes[-1]

        return costs[str(v_class)]
