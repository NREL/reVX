************************************************************************
Welcome to the `reV <https://nrel.github.io/reV>`_ eXchange (reVX) tool!
************************************************************************

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

.. image:: https://codecov.io/gh/nrel/reVX/branch/main/graph/badge.svg?token=3J5M44VAA9
    :target: https://codecov.io/gh/nrel/reVX

.. image:: https://zenodo.org/badge/201337735.svg
   :target: https://zenodo.org/badge/latestdoi/201337735

.. inclusion-intro

reVX command line tools
=======================

- `reVX <https://nrel.github.io/reVX/_cli/reVX.html#revx>`_
- `reV-ReEDS <https://nrel.github.io/reVX/_cli/reV-ReEDS.html#rev-reeds>`_
- `reV-rpm <https://nrel.github.io/reVX/_cli/reV-rpm.html#rev-rpm>`_
- `reV-plexos <https://nrel.github.io/reVX/_cli/reVX-plexos.html#rev-plexos>`_
- `plexos-plants <https://nrel.github.io/reVX/_cli/plexos-plants.html#plexos-plants>`_
- `mean-wind-dirs <https://nrel.github.io/reVX/_cli/mean-wind-dirs.html#mean-wind-dirs>`_
- `prominent-wind-dirs <https://nrel.github.io/reVX/_cli/prominent-wind-dirs.html#prominent-wind-dirs>`_
- `setbacks <https://nrel.github.io/reVX/_cli/setbacks.html#setbacks>`_
- `offshore-assembly-areas <https://nrel.github.io/reVX/_cli/offshore-assembly-areas.html#offshore-assembly-areas>`_
- `offshore-dist-to-ports <https://nrel.github.io/reVX/_cli/offshore-dist-to-ports.html#offshore-dist-to-ports>`_
- `offshore-inputs <https://nrel.github.io/reVX/_cli/offshore-inputs.html#offshore-inputs>`_

Using Eagle Env
===============

If you would like to run reVX on Eagle (NREL's HPC) you can use a pre-compiled
conda env:

.. code-block:: bash

    conda activate /shared-projects/rev/modulefiles/conda/envs/rev/

or

.. code-block:: bash

    source activate /shared-projects/rev/modulefiles/conda/envs/rev/

Installing reVX
===============

NOTE: The installation instruction below assume that you have python installed
on your machine and are using `conda <https://docs.conda.io/en/latest/index.html>`_
as your package/environment manager.

Option 1: Install from PIP or Conda (recommended for analysts):
---------------------------------------------------------------

1. Create a new environment:
    ``conda create --name revx python=3.7``

2. Activate directory:
    ``conda activate revx``

3. Install reVX:
    1) ``pip install NREL-reVX`` or
    2) ``conda install nrel-revx --channel=nrel``

        - NOTE: The best guarantee you will have of a correct installation is by following the same `installation process that we use for our automated test suite <https://github.com/NREL/reVX/blob/7932a095c222e2e5c70bfc7b4813a68a1da2493a/.github/workflows/pull_request_tests.yml#L29-L33>`_.
        
        - NOTE: If you install using conda and you want to use:
            * `HSDS <https://github.com/NREL/hsds-examples>`_ you will also need to install h5pyd manually: ``pip install h5pyd``
            * `Turbine Flicker <https://nrel.github.io/reVX/_cli/reVX.turbine_flicker.turbine_flicker.html>`_ you will need to install `HOPP <https://github.com/nrel/HOPP>`_ manually: ``pip install HOPP``

        - NOTE: If you install using pip and want to run `exclusion setbacks <https://nrel.github.io/reVX/_cli/reVX.setbacks.setbacks.html>`_ you will need to install rtree manually:
            * ``conda install rtree``
            * `pip installation instructions <https://pypi.org/project/Rtree/#:~:text=Rtree%20is%20a%20ctypes%20Python,Multi%2Ddimensional%20indexes>`_

Option 2: Clone repo (recommended for developers)
-------------------------------------------------

1. from home dir, ``git clone git@github.com:NREL/reVX.git``

2. Create ``reVX`` environment and install package
    1) Create a conda env: ``conda create -n revx``
    2) Run the command: ``conda activate revx``
    3) cd into the repo cloned in 1.
    4) | prior to running ``pip`` below, make sure the branch is correct (install
       | from main!)
    5) | Install ``reVX`` and its dependencies by running:
       | ``pip install .`` (or ``pip install -e .`` if running a dev branch
       | or working on the source code)

3. Check that ``reVX`` was installed successfully
    1) | From any directory, run the following commands. This should return the
       | help pages for the CLI's.

        - ``reVX``


Recommended Citation
====================

Update with current version and DOI:

Michael Rossol, Grant Buster, and Robert Spencer. The Renewable Energy
Potential(V) eXchange Tool: reVX. https://github.com/NREL/reVX
(version v0.3.20), 2021. https://doi.org/10.5281/zenodo.4507580.
