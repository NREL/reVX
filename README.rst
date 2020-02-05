reVX
####
Renewable Energy Potential(V) eXchange Tool: reVX

reVX command line tools
***********************

- ``reVX``
- ``reV-ReEDS``
- `reV-rpm <https://github.com/NREL/reVX/tree/master/examples/rpm/README.rst>`_
- ``NSRDB``
- ``WIND``

Using Eagle Module
******************

If you would like to run reV on Eagle (NREL's HPC) you can use a pre-compiled module:
::
    module use /shared-projects/rev/modulefiles
    module load reVX

Setup to run revX
*****************

Option 1: PIP Install the most recent version of master (recommended for analysts):
===================================================================================

1. Create a new environment: ``conda create --name revx python=3.7``

2. Activate directory: ``conda activate revx``

3. Install reV which is a required dependency: ``pip install git+ssh://git@github.com/NREL/reV.git`` or ``pip install git+https://github.com/NREL/reV.git``

4. Install reVX: ``pip install git+ssh://git@github.com/NREL/reVX.git`` or ``pip install git+https://github.com/NREL/reVX.git``

Option 2: Clone repo (recommended for developers)
=================================================

1. from home dir, ``git clone https://github.com/NREL/reVX.git``
    1) enter github username
    2) enter github password

2. Install reVX environment and modules (using conda)
    1) cd into reVX repo cloned above
    2) cd into ``bin/$OS/``
    3) run the command: ``conda env create -f revx.yml``. If conda can't find any packages, try removing them from the yml file.
    4) run the command: ``conda activate revx``
    5) prior to running ``pip`` below, make sure branch is correct (install from master!)
    6) cd back to the reVX repo (where setup.py is located)
    7) install pre-commit: ``pre-commit install``
    8) run ``pip install .`` (or ``pip install -e .`` if running a dev branch or working on the source code)

3. from home dir, ``git clone https://github.com/NREL/reV.git``
    1) enter github username
    2) enter github password

4. Install reV environment and modules (using conda)
    1) cd into reV repo cloned above
    2) prior to running ``pip`` below, make sure branch is correct (install from master!)
    3) cd back to the reV repo (where setup.py is located)
    4) install pre-commit: ``pre-commit install``
    5) run ``pip install .`` (or ``pip install -e .`` if running a dev branch or working on the source code)

3. Check that rev was installed successfully
    1) From any directory, run the following commands. This should return the help pages for the CLI's.
        - ``reVX``
        - ``reV-rpm``
