'''
Load tables
-----------

gb_msoas : a table of MSOAs that were included when fitting the model.
           The following columns are included:
               - msoa / msoa_name : The code / name of the MSOAs
               - lad / lad_name : The code / name of the local authority district (LAD)
               - region / region_name : The code / name of the region. Note that there is no
                                        equivalent to regions for Wales and Scotland so the
                                        lad / lad_name is used instead.
               - country
'''

from importlib import resources as impresources

import pandas as pd
from . import data

table_path = impresources.files(data) / 'msoa_table.parquet'
gb_msoas = pd.read_parquet(table_path)
