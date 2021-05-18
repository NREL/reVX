import json
import os

CONFIGDIR = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(CONFIGDIR, 'power_to_voltage.json'), 'rt') as f:
    power_to_voltage = json.load(f)

with open(os.path.join(CONFIGDIR, 'base_line_costs.json'), 'rt') as f:
    base_line_costs = json.load(f)

with open(os.path.join(CONFIGDIR, 'multipliers.json'), 'rt') as f:
    multipliers = json.load(f)

with open(os.path.join(CONFIGDIR, 'transformer_costs.json'), 'rt') as f:
    transformer_costs = json.load(f)
