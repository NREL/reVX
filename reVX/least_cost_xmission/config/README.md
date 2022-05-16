Configuration JSON files with costs, etc. for lines, substations, and
transformers.  There was missing data in most tables. Any missing data was
filled with the average value for that data group.

The following ISOs are used in the multipliers and base line cost files.
- TEPPC
- SCE
- MISO
- Southeast

power\_to\_voltage:
    Line capacity to voltage conversion table. Keys are capacity in MW, values
    are voltages in kV. Note that there are two capacities for 230kV lines and
    should be two sets of costs in the cost tables.
iso_multipliers
    Baseline cost multipliers for ISOs. Slopes are in %, e.g. "2" = 2%.

transformer_costs:
     Transformer costs in $/MVA. First key is tie line voltage, second key is
     existing t-line voltage, but that doesn't matter much as the data is
     symmetrical. From MISO table.

base\_line_costs:
    Costs for lines in 2019 US$ / mile. Keys are max line capacity in MW.
    Note that TEPPC does not include ROW in base line cost, but all other ISOs
    do.

power_classes:
    Keys are reV power classes. Values are appropriate line capacities for
    that class.

power\_to_voltage:
    Keys are line capacities in MW. Values are line voltages in kV.
