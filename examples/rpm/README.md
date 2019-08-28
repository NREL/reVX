# reV to RPM Pipeline

The reV to RPM pipeline can be fun from the command line using the `reV-RPM`  
command line interface (CLI), or using the rpm module.

## reV-RPM CLI

The `reV-RPM` command line tools can be used to run the reV to RPM pipeline  
using the default clustering and representative profile options.

There are 3 steps to the reV to RPM pipeline:
    1) Clustering of the capacity factor (CF) profiles within each RPM region
    2) Applying geo-spatial exclusions to the CF profiles (optional)
    3) Extracting representative profiles from each region

The entire pipeline can be run using the following CLI call:
