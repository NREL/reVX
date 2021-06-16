import json
import os

CONFIGDIR = os.path.dirname(os.path.realpath(__file__))

# Cost multipliers for medium and short lines
SHORT_MULT = 1.5
MEDIUM_MULT = 1.2

# Cut offs are originally in miles but are converted to kilometers
SHORT_CUTOFF = 3*5280/3.28084/1000
MEDIUM_CUTOFF = 10*5280/3.28084/1000

CELL_SIZE = 90  # meters, size of cell. Both dims must be equal
TEMPLATE_SHAPE = (33792, 48640)

# Decimal % distance to buffer clipped cost raster by. This helps to find the
# cheapest path. Larger values will run slower
CLIP_RASTER_BUFFER = 0.05

# Number of load centers and sinks to connect to
NUM_LOAD_CENTERS = 1
NUM_SINKS = 1

# Number of times to report on progress of SC point processing, e.g. 5 means
# about every 20%
REPORTING_STEPS = 10

# Costs multiplier for cells affected by transmission barriers
BARRIERS_MULT = 100

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

with open(os.path.join(CONFIGDIR, 'min_power_classes.json'), 'rt') as f:
    min_power_classes = json.load(f)

with open(os.path.join(CONFIGDIR, 'new_substation_costs.json'), 'rt') as f:
    new_sub_costs = json.load(f)

with open(os.path.join(CONFIGDIR, 'upgrade_substation_costs.json'), 'rt') as f:
    upgrade_sub_costs = json.load(f)

# TODO - check that the iso regions in the cost files match the iso regions
