# -*- coding: utf-8 -*-
"""
setup.py
"""
import os
from codecs import open
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from subprocess import check_call
import shlex
from warnings import warn

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    readme = f.read()

with open(os.path.join(here, "reVX", "version.py"), encoding="utf-8") as f:
    version = f.read()

version = version.split('=')[-1].strip().strip('"').strip("'")


class PostDevelopCommand(develop):
    """
    Class to run post setup commands
    """

    def run(self):
        """
        Run method that tries to install pre-commit hooks
        """
        try:
            check_call(shlex.split("pre-commit install"))
        except Exception as e:
            warn("Unable to run 'pre-commit install': {}"
                 .format(e))

        develop.run(self)


with open("requirements.txt") as f:
    install_requires = f.readlines()

test_requires = ["pytest>=5.2", ]
description = ("National Renewable Energy Laboratory's (NREL's) Renewable "
               "Energy Potential(V) eXchange Tool: reVX")

setup(
    name="NREL-reVX",
    version=version,
    description=description,
    long_description=readme,
    author="Michael Rossol",
    author_email="michael.rossol@nrel.gov",
    url="https://nrel.github.io/reVX/",
    packages=find_packages(),
    package_dir={"reVX": "reVX"},
    entry_points={
        "console_scripts": ["reVX=reVX.cli:main",
                            "reV-ReEDS=reVX.reeds.reeds_cli:main",
                            "reV-rpm=reVX.rpm.rpm_cli:main",
                            "reV-plexos=reVX.plexos.rev_reeds_plexos_cli:main",
                            "plexos-plants=reVX.plexos.plexos_plants_cli:main",
                            "setbacks=reVX.setbacks.setbacks_cli:main",
                            ("prominent-wind-dirs=reVX.wind_dirs."
                             "prominent_wind_dirs_cli:main"),
                            ("mean-wind-dirs=reVX.wind_dirs."
                             "mean_wind_dirs_cli:main"),
                            ("offshore-dist-to-ports=reVX.offshore."
                             "dist_to_ports_cli:main"),
                            ("offshore-assembly-areas=reVX.offshore."
                             "assembly_areas_cli:main"),
                            ("offshore-inputs=reVX.offshore."
                             "offshore_inputs_cli:main"),
                            ("turbine-flicker=reVX.turbine_flicker."
                             "turbine_flicker_cli:main"),
                            ("simple-plant-builder=reVX.plexos."
                             "simple_plant_builder_cli:main"),
                            ("xmission-cost-creator=reVX.least_cost_xmission."
                             "cost_creator_cli:main"),
                            ("least-cost-xmission=reVX.least_cost_xmission."
                             "least_cost_xmission_cli:main"),
                            ("least-cost-paths=reVX.least_cost_xmission."
                             "least_cost_paths_cli:main")
                            ],
    },
    include_package_data=True,
    license="BSD 3-Clause",
    zip_safe=False,
    keywords="reVX",
    python_requires='>=3.7,<3.10',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    test_suite="tests",
    install_requires=install_requires,
    extras_require={
        "test": test_requires,
        "dev": test_requires + ["flake8", "pre-commit", "pylint"],
    },
    cmdclass={"develop": PostDevelopCommand},
)
