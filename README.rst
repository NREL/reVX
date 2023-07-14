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

.. image:: https://codecov.io/gh/nrel/reVX/branch/main/graph/badge.svg?token=3J5M44VAA9
    :target: https://codecov.io/gh/nrel/reVX

.. image:: https://zenodo.org/badge/201337735.svg
   :target: https://zenodo.org/badge/latestdoi/201337735

.. inclusion-intro

reVX command line tools
=======================

- `reVX <https://nrel.github.io/reVX/_cli/reVX.html#revx>`_
- `reV-rpm <https://nrel.github.io/reVX/_cli/reV-rpm.html#rev-rpm>`_
- `reV-plexos <https://nrel.github.io/reVX/_cli/reV-plexos.html>`_
- `plexos-plants <https://nrel.github.io/reVX/_cli/plexos-plants.html#plexos-plants>`_
- `mean-wind-dirs <https://nrel.github.io/reVX/_cli/mean-wind-dirs.html#mean-wind-dirs>`_
- `prominent-wind-dirs <https://nrel.github.io/reVX/_cli/prominent-wind-dirs.html#prominent-wind-dirs>`_
- `setbacks <https://nrel.github.io/reVX/_cli/setbacks.html#setbacks>`_
- `offshore-assembly-areas <https://nrel.github.io/reVX/_cli/offshore-assembly-areas.html#offshore-assembly-areas>`_
- `offshore-dist-to-ports <https://nrel.github.io/reVX/_cli/offshore-dist-to-ports.html#offshore-dist-to-ports>`_
- `offshore-inputs <https://nrel.github.io/reVX/_cli/offshore-inputs.html#offshore-inputs>`_

Installing reVX
===============

NOTE: The installation instruction below assume that you have python installed
on your machine and are using `conda <https://docs.conda.io/en/latest/index.html>`_
as your package/environment manager.

#. Create a new environment:
    ``conda create --name revx python=3.8``

#. Activate your new environment:
    ``conda activate revx``

#. Clone the repo:
    From your home directory ``/home/{user}/`` or another directory that you have permissions in, run the command ``git clone git@github.com:NREL/reVX.git`` and then go into your cloned repository: ``cd reVX``

#. Install reVX:
    1) Follow the installation commands installation process that we use for our automated test suite `here <https://github.com/NREL/reVX/blob/main/.github/workflows/pull_request_tests.yml#L31-L34>`_. Make sure that you call ``pip install -e .`` from within the cloned repository directory e.g. ``/home/{user}/reVX/``

        - NOTE: If you install using pip and want to run `exclusion setbacks <https://nrel.github.io/reVX/_cli/reVX.setbacks.setbacks.html>`_ you will need to install rtree manually:
            * ``conda install rtree``
            * `pip installation instructions <https://pypi.org/project/Rtree/#:~:text=Rtree%20is%20a%20ctypes%20Python,Multi%2Ddimensional%20indexes>`_

Recommended Citation
====================

Update with current version and DOI:

Michael Rossol, Grant Buster, and Robert Spencer. The Renewable Energy
Potential(V) eXchange Tool: reVX. https://github.com/NREL/reVX
(version v0.3.20), 2021. https://doi.org/10.5281/zenodo.4507580.
