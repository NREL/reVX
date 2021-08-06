"""
Load default configuration for tie-line cost determination

Mike Bannister
7/27/21
"""
import os

from rex.utilities.utilities import safe_json_load

D_DIR = os.path.dirname(os.path.realpath(__file__))


class XmissionConfig(dict):
    """ Load default tie lines configuration """

    def __init__(self, power_to_voltage_fpath=None, base_line_costs_fpath=None,
                 iso_mults_fpath=None, transformer_costs_fpath=None,
                 iso_lookup_fpath=None, power_classes_fpath=None,
                 min_power_classes_fpath=None, new_sub_costs_fpath=None,
                 upgrade_sub_costs_fpath=None):
        """
        Use default values for all parameters if None.

        Parameters
        ----------
        power_to_voltage_fpath : None | str
            JSON file linking line voltages with capacities.
        base_line_costs_fpath : None | str
            Base line cost per mile for ISOs file. Use default if None
        iso_mults_fpath : None | str
            ISO multipliers JSON file. Use default if None
        transformer_costs_fpath: None | str
            JSON file with transformer costs.
        iso_lookup_fpath : None | str
            JSON file linking ISO region code to name. Use default if None
        power_classes_fpath : None | str
            JSON of line capacities to use for reV power classes.
        min_power_classes_fpath : None | str
            JSON file with actual line capacities to use for reV power classes.
        new_sub_costs_fpath : None | str
            JSON file with existing new substation construction costs
        upgrade_sub_costs_fpath : None | str
            JSON file with existing substation upgrade costs
        """
        super().__init__()

        if power_to_voltage_fpath is None:
            power_to_voltage_fpath = os.path.join(D_DIR,
                                                  'power_to_voltage.json')
        self['power_to_voltage'] = safe_json_load(power_to_voltage_fpath)

        if base_line_costs_fpath is None:
            base_line_costs_fpath = os.path.join(D_DIR, 'base_line_costs.json')
        self['base_line_costs'] = safe_json_load(base_line_costs_fpath)

        if iso_mults_fpath is None:
            iso_mults_fpath = os.path.join(D_DIR, 'iso_multipliers.json')
        self['iso_mults'] = safe_json_load(iso_mults_fpath)

        if transformer_costs_fpath is None:
            transformer_costs_fpath = os.path.join(D_DIR,
                                                   'transformer_costs.json')
        self['transformer_costs'] = safe_json_load(transformer_costs_fpath)

        if iso_lookup_fpath is None:
            iso_lookup_fpath = os.path.join(D_DIR, 'iso_lookup.json')
        self['iso_lookup'] = safe_json_load(iso_lookup_fpath)

        if power_classes_fpath is None:
            power_classes_fpath = os.path.join(D_DIR, 'power_classes.json')
        self['power_classes'] = safe_json_load(power_classes_fpath)

        if min_power_classes_fpath is None:
            min_power_classes_fpath = os.path.join(D_DIR,
                                                   'min_power_classes.json')
        self['min_power_classes'] = safe_json_load(min_power_classes_fpath)

        if new_sub_costs_fpath is None:
            new_sub_costs_fpath = os.path.join(D_DIR,
                                               'new_substation_costs.json')
        self['new_sub_costs'] = safe_json_load(new_sub_costs_fpath)

        if upgrade_sub_costs_fpath is None:
            upgrade_sub_costs_fpath = os.path.join(D_DIR,
                                                   'upgrade_substation_costs.'
                                                   'json')
        self['upgrade_sub_costs'] = safe_json_load(upgrade_sub_costs_fpath)