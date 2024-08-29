import pandas as pd
from importlib import resources as impresources
from . import data

table_path = impresources.files(data) / 'msoa_table.parquet'
gb_msoas = pd.read_parquet(table_path)
