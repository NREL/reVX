****
reVX
****

.. image:: https://github.com/NREL/reVX/workflows/Documentation/badge.svg
    :target: https://nrel.github.io/reVX/

.. image:: https://github.com/NREL/reVX/workflows/Pytests/badge.svg
    :target: https://github.com/NREL/reVX/actions?query=workflow%3A%22Pytests%22

.. image:: https://github.com/NREL/reVX/workflows/Lint%20Code%20Base/badge.svg
    :target: https://github.com/NREL/reVX/actions?query=workflow%3A%22Lint+Code+Base%22

.. image:: https://img.shields.io/pypi/pyversions/NREL-reVX.svg
    :target: https://pypi.org/project/NREL-reVX/

.. image:: https://badge.fury.io/py/NREL-reVX.svg
    :target: https://badge.fury.io/py/NREL-reVX

.. image:: https://anaconda.org/nrel/nrel-revx/badges/version.svg
    :target: https://anaconda.org/nrel/nrel-revx

.. image:: https://anaconda.org/nrel/nrel-revx/badges/license.svg
    :target: https://anaconda.org/nrel/nrel-revx

.. image:: https://codecov.io/gh/nrel/reVX/branch/master/graph/badge.svg?token=3J5M44VAA9
    :target: https://codecov.io/gh/nrel/reVX


Renewable Energy Potential(V) eXchange Tool: reVX

.. inclusion-intro

reVX command line tools
=======================

- `reVX <https://nrel.github.io/reVX/reVX/reVX.cli.html#revx>`_
- `reV-ReEDS <https://nrel.github.io/reVX/reVX/reVX.reeds.reeds_cli.html#rev-reeds>`_
- `reV-rpm <https://nrel.github.io/reVX/reVX/reVX.rpm.rpm_cli.html#rev-rpm>`_
- `reV-plexos <https://nrel.github.io/reVX/reVX/reVX.plexos.rev_reeds_plexos_cli.html#rev-plexos>`_
- `plexos-plants <https://nrel.github.io/reVX/reVX/reVX.plexos.plexos_plants_cli.html#plexos-plants>`_
- `wind-dirs <https://nrel.github.io/reVX/reVX/reVX.wind_dirs.wind_dirs_cli.html#wind-dirs>`_
- `wind-setbacks <https://nrel.github.io/reVX/reVX/reVX.wind_setbacks.wind_setbacks_cli.html#wind-setbacks>`_

Using Eagle Env / Module
========================

If you would like to run reVX on Eagle (NREL's HPC) you can use a pre-compiled
conda env:

.. code-block:: bash

    conda activate /shared-projects/rev/modulefiles/conda_env

or

.. code-block:: bash

    source activate /shared-projects/rev/modulefiles/conda_env

or module:

.. code-block:: bash

    module use /shared-projects/rev/modulefiles
    module load reVX

**NOTE: Loading the reVX module can take several minutes**

Installing reVX
=================

Option 1: Install from PIP or Conda (recommended for analysts):
---------------------------------------------------------------

1. Create a new environment:
    ``conda create --name revx python=3.7``

2. Activate directory:
    ``conda activate revx``

3. Install reVX:
    1) ``pip install NREL-reVX`` or
    2) ``conda install nrel-revx --channel=nrel``

        - NOTE: If you install using conda and want to use `HSDS <https://github.com/NREL/hsds-examples>`_
          you will also need to install h5pyd manually: ``pip install h5pyd``

        - NOTE: If you install using pip and want to run `Wind Setbacks <https://nrel.github.io/reVX/reVX/reVX.wind_setbacks.wind_setbacks.html>_` you will need to install rtree manually:
            * ``conda install rtree``
            * `pip installation instructions <https://pypi.org/project/Rtree/#:~:text=Rtree%20is%20a%20ctypes%20Python,Multi%2Ddimensional%20indexes>`_

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
