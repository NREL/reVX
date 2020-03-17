reV to ReEDS Pipeline
=====================

The reV to ReEDS pipeline can be run from the command line using the
`reV-ReEDS` command line interface (CLI), or using the reeds sub-package.

reV-ReEDS CLI
-------------

The ``reV-ReEDS`` command line tools can be used to run the reV to ReEDS
pipeline using the default clustering and representative profile options.

There are 3 steps to the reV to ReEDS pipeline:

    1) Classifying the reV 'resource' into 'regions', 'classes', and 'bins'
    2) Extracting representative profiles from the above regions
    3) Extract the ReEDS 'timeslices' from the repesentative profiles

The entire pipeline can be run using the following CLI call:

.. code-block:: bash

    reV-ReEDS --name="Job_name" local --out_dir="./output_directory" classify --rev_table="$TESTDATADIR/reV_sc/sc_table.csv" --resource_classes="$TESTDATADIR/reeds/inputs/reeds_class_bins.csv" profiles --cf_profiles="%TESTDATADIR/reV_gen/gen_pv_2012.h5" timeslices --timeslices="$TESTDATADIR/reeds/inputs/timeslices.csv"

Results of each step of the process will be written to ``out_dir``. Note that
additional options are available for each step of the process: ``classify``,
``profiles``, and ``timeslices``.

Each step can be run individually:

.. code-block:: bash

    reV-ReEDS --name="Job_name" local --out_dir="./output_directory" classify --rev_table="$TESTDATADIR/reV_sc/sc_table.csv" --resource_classes="$TESTDATADIR/reeds/inputs/reeds_class_bins.csv"

.. code-block:: bash

    reV-ReEDS --name="Job_name" local --out_dir="./output_directory" profiles --reeds_table="./output_directory/Job_name_supply_curve_raw_full.csv --cf_profiles="%TESTDATADIR/reV_gen/gen_pv_2012.h5"

NOTE: when running profiles without classify the ReEDS classification table
must be supplied.

.. code-block:: bash

    reV-ReEDS --name="Job_name" local --out_dir="./output_directory" timeslices --profiles="./output_directory/Job_name_hourly_cf.h5--timeslices="$TESTDATADIR/reeds/inputs/timeslices.csv"

NOTE: when running timeslices without profiles the ReEDS representative
profiles must be supplied.

ReEDS sub-package
-----------------
