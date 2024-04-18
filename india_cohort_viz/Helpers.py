# imports
import os
import json

import pandas as pd

from pandas.io.stata import StataReader


def data_loader(path_to_stata_file:str, cwd:str)->pd.DataFrame:

    path_to_cols = os.path.join(
        cwd, os.path.join('auxiliar_data', 'columns.JSON')
    )

    with open(path_to_cols, 'r') as file:
        cols_dict = json.load(file)

    stata_reader = StataReader(path_to_stata_file)

    df = stata_reader.read(preserve_dtypes=False, convert_categoricals=False, convert_dates=True)

    df = df[cols_dict.keys()].copy()

    df.columns = [cols_dict[key] for key in df.columns]

    return df
