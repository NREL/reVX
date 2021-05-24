import json
import os

CONFIGDIR = os.path.dirname(os.path.realpath(__file__))

# Cost multipliers for medium and short lines
SHORT_MULT = 1.5
MEDIUM_MULT = 1.2

# Cut offs are originally in miles but are converted to meters
SHORT_CUTOFF = 3*5280/3.28084
MEDIUM_CUTOFF = 10*5280/3.28084

CELL_SIZE = 90  # meters, size of cell. Both dims must be equal
TEMPLATE_SHAPE = (33792, 48640)


# Load json files
with open(os.path.join(CONFIGDIR, 'power_to_voltage.json'), 'rt') as f:
    power_to_voltage = json.load(f)

with open(os.path.join(CONFIGDIR, 'base_line_costs.json'), 'rt') as f:
    base_line_costs = json.load(f)

with open(os.path.join(CONFIGDIR, 'multipliers.json'), 'rt') as f:
    iso_mults = json.load(f)

with open(os.path.join(CONFIGDIR, 'transformer_costs.json'), 'rt') as f:
    transformer_costs = json.load(f)

with open(os.path.join(CONFIGDIR, 'iso_lookup.json'), 'rt') as f:
    iso_lookup = json.load(f)

with open(os.path.join(CONFIGDIR, 'power_classes.json'), 'rt') as f:
    power_classes = json.load(f)
