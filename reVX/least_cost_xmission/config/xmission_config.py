import os
import json

D_DIR = os.path.dirname(os.path.realpath(__file__))


class XmissionConfig:
    """ Load default tie lines configuration """

    def __init__(self, power_to_voltage_f=None, base_line_costs_f=None,
                 iso_mults_f=None, transformer_costs_f=None,
                 iso_lookup_f=None, power_classes_f=None,
                 min_power_classes_f=None, new_substation_costs_f=None,
                 upgrade_sub_costs_f=None):
        """
        TODO
        """
        if power_to_voltage_f is None:
            power_to_voltage_f = os.path.join(D_DIR, 'power_to_voltage.json')
        with open(power_to_voltage_f, 'rt') as f:
            self.power_to_voltage = json.load(f)

        if base_line_costs_f is None:
            base_line_costs_f = os.path.join(D_DIR, 'base_line_costs.json')
        with open(base_line_costs_f, 'rt') as f:
            self.base_line_costs = json.load(f)

        if iso_mults_f is None:
            iso_mults_f = os.path.join(D_DIR, 'iso_multipliers.json')
        with open(iso_mults_f, 'rt') as f:
            self.iso_mults = json.load(f)

        if transformer_costs_f is None:
            transformer_costs_f = os.path.join(D_DIR, 'transformer_costs.json')
        with open(transformer_costs_f, 'rt') as f:
            self.transformer_costs = json.load(f)

        if iso_lookup_f is None:
            iso_lookup_f = os.path.join(D_DIR, 'iso_lookup.json')
        with open(iso_lookup_f, 'rt') as f:
            self.iso_lookup = json.load(f)

        if power_classes_f is None:
            power_classes_f = os.path.join(D_DIR, 'power_classes.json')
        with open(power_classes_f, 'rt') as f:
            self.power_classes = json.load(f)

        if min_power_classes_f is None:
            min_power_classes_f = os.path.join(D_DIR, 'min_power_classes.json')
        with open(min_power_classes_f, 'rt') as f:
            self.min_power_classes = json.load(f)

        if new_substation_costs_f is None:
            new_substation_costs_f = os.path.join(D_DIR,
                                                  'new_substation_costs.json')
        with open(new_substation_costs_f, 'rt') as f:
            self.new_sub_costs = json.load(f)

        if upgrade_sub_costs_f is None:
            upgrade_sub_costs_f = os.path.join(D_DIR,
                                               'upgrade_substation_costs.json')
        with open(upgrade_sub_costs_f, 'rt') as f:
            self.upgrade_sub_costs = json.load(f)
