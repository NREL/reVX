Configuration JSON files with costs, etc. for lines, substations, and
transformers.  There was missing data in most tables. Any missing data was
filled with the average value for that data group.

The following ISOs are used in the multipliers and base line cost files.
- TEPPC
- SCE
- MISO
- Southeast

power_to_voltage.json:
    Line capacity to voltage conversion table. Keys are capacity in MW, values
    are voltages in kV. Note that there are two capacities for 230kV lines and
    should be two sets of costs in the cost tables.

multipliers.json
    Baseline cost multipliers for ISOs. Slopes are in %, e.g. "2" = 2%.

transformer_costs.json:
     Transformer costs in $/MVA. First key is tie line voltage, second key is
     existing t-line voltage, but that doesn't matter much as the data is
     symmetrical. From MISO table.

base_line_costs.json:
    Costs for lines in US$ / mile. Keys are max line capacity in MW. Note that
    TEPPC does not include ROW in base line cost, but all other ISOs do.

power_classes.json:
    Keys are reV power classes. Values are appropriate line capacities for
    that class.

power_to_voltage.json:
    Keys are line capacities in MW. Values are line voltages in kV.