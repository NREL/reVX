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

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/NREL/reVX/master

.. image:: https://zenodo.org/badge/201337735.svg
   :target: https://zenodo.org/badge/latestdoi/201337735

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

Using Eagle Env
===============

If you would like to run reVX on Eagle (NREL's HPC) you can use a pre-compiled
conda env:

.. code-block:: bash

    conda activate /shared-projects/rev/modulefiles/conda_env

or

.. code-block:: bash

    source activate /shared-projects/rev/modulefiles/conda_env

.. or module:

.. .. code-block:: bash

..     module use /shared-projects/rev/modulefiles
..     module load reVX

.. **NOTE: Loading the reVX module can take several minutes**

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

2. Create ``reVX`` environment and install package
    1) Create a conda env: ``conda create -n revx``
    2) Run the command: ``conda activate revx``
    3) cd into the repo cloned in 1.
    4) prior to running ``pip`` below, make sure the branch is correct (install
       from master!)
    5) Install ``reVX`` and its dependencies by running:
       ``pip install .`` (or ``pip install -e .`` if running a dev branch
       or working on the source code)

3. Check that ``reVX`` was installed successfully
    1) From any directory, run the following commands. This should return the
       help pages for the CLI's.

        - ``reVX``


Recommended Citation
====================

Update with current version and DOI:

Michael Rossol, Grant Buster, and Robert Spencer. The Renewable Energy Potential(V) eXchange Tool: reVX. https://github.com/NREL/reVX (version v0.3.20), 2021. https://doi.org/10.5281/zenodo.4507580.
