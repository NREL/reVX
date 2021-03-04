# -*- coding: utf-8 -*-
"""reVX PLEXOS unit test module
"""
from click.testing import CliRunner
import numpy as np
import json
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import shutil
import tempfile
import traceback

from rex import Resource
from rex.utilities.loggers import LOGGERS

from reVX.plexos.rev_reeds_plexos import SimplePlantBuilder, PlexosNode
from reVX.plexos.rev_reeds_plexos_cli import main
from reVX import TESTDATADIR

REV_SC = os.path.join(TESTDATADIR, 'reV_sc/wtk_coe_2017_cem_v3_wind_conus_multiyear_colorado.csv')

CF_FPATH = os.path.join(TESTDATADIR,
                        'reV_gen/naris_rev_wtk_gen_colorado_2007.h5')

# import and rename sc table to pretend its a modern rev2 sc table output
REV_SC = pd.read_csv(REV_SC)
REV_SC = REV_SC.rename({'gid': 'sc_gid',
                        'resource_ids': 'res_gids',
                        'resource_ids_cnts': 'gid_counts',
                        'lat': 'latitude',
                        'lon': 'longitude',
                        'ncf': 'mean_cf',
                        }, axis=1)


# these points are chosen based on whats available in the CF_FPATH
PLANT_META = pd.DataFrame({'latitude': [37.24, 37.24, 40.8],
                           'longitude': [-102.5, -102.49, -105.7],
                           'capacity': [100, 100, 50]})


if __name__ == '__main__':
    # test that it can handle some of these columns as lists
    REV_SC['res_gids'] = REV_SC['res_gids'].apply(json.loads)

    pb = SimplePlantBuilder(PLANT_META, REV_SC, CF_FPATH)

    assert isinstance(pb._sc_table['gid_counts'].values[0], list)
    assert isinstance(pb._sc_table['gid_capacity'].values[0], list)
    assert isinstance(pb._sc_table['gid_counts'].values[0][0], int)
    assert isinstance(pb._sc_table['gid_capacity'].values[0][0], float)


    pb = SimplePlantBuilder.run(PLANT_META, REV_SC, CF_FPATH, max_workers=1)
