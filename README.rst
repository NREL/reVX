reVX
****

.. image:: https://badge.fury.io/py/NREL-reVX.svg
    :target: https://badge.fury.io/py/NREL-reVX

.. image:: https://github.com/NREL/reVX/workflows/Documentation/badge.svg
    :target: https://nrel.github.io/reVX/

.. image:: https://github.com/NREL/reVX/workflows/Pytests/badge.svg
    :target: https://github.com/NREL/reVX/actions?query=workflow%3A%22Pytests%22

Renewable Energy Potential(V) eXchange Tool: reVX

.. inclusion-intro

reVX command line tools
=======================

- `reVX <https://nrel.github.io/reVX/reVX/reVX.cli.html#revx>`_
- `reV-ReEDS <https://nrel.github.io/reVX/reVX/reVX.reeds.reeds_cli.html#rev-reeds>`_
- `reV-rpm <https://nrel.github.io/reVX/reVX/reVX.rpm.rpm_cli.html#rev-rpm>`_
- `reV-plexos <https://nrel.github.io/reVX/reVX/reVX.plexos.plexos_cli.html#rev-plexos>`_
- `wind-dirs <https://nrel.github.io/reVX/reVX/reVX.wind_dirs.wind_dirs_cli.html#wind-dirs>`_

Using Eagle Module
==================

If you would like to run reV on Eagle (NREL's HPC) you can use a pre-compiled
module:

.. code-block:: bash

    module use /shared-projects/rev/modulefiles
    module load reVX

Installing reVX
=================

Option 1: PIP Install (recommended for analysts):
-------------------------------------------------

1. Create a new environment:
    ``conda create --name revx python=3.7``

2. Activate directory:
    ``conda activate revx``

3. Install reVX:
    ``pip install NREL-reVX``

Option 2: Clone repo (recommended for developers)
-------------------------------------------------

1. from home dir, ``git clone https://github.com/NREL/reVX.git``
    1) enter github username
    2) enter github password

2. Install reVX environment and modules (using conda)
    1) cd into reVX repo cloned above
    2) cd into ``bin/$OS/``
    3) run the command: ``conda env create -f revx.yml``. If conda can't find
       any packages, try removing them from the yml file.

    4) run the command: ``conda activate revx``
    5) prior to running ``pip`` below, make sure branch is correct (install
       from master!)

    6) cd back to the reVX repo (where setup.py is located)
    7) install pre-commit: ``pre-commit install``
    8) run ``pip install .`` (or ``pip install -e .`` if running a dev branch
       or working on the source code)

3. from home dir, ``git clone https://github.com/NREL/reV.git``
    1) enter github username
    2) enter github password

4. Install reV environment and modules (using conda)
    1) cd into reV repo cloned above
    2) prior to running ``pip`` below, make sure branch is correct (install
       from master!)

    3) cd back to the reV repo (where setup.py is located)
    4) install pre-commit: ``pre-commit install``
    5) run ``pip install .`` (or ``pip install -e .`` if running a dev branch
       or working on the source code)

3. Check that reV and reVX were installed successfully
    1) From any directory, run the following commands. This should return the
       help pages for the CLI's.

        - ``reVX``
        - ``reV``
