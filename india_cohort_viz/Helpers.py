# imports
import os
import json

import pandas as pd
import streamlit as st

from india_cohort_viz.Summary import summary_for_continuos, summary_for_categorical
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

def show_data(radio_val, data:pd.DataFrame, continuos:list=[], cat_class:list=[]):

    if radio_val in continuos:
        general_summary = summary_for_continuos(data, stat_col='Status', var_col=radio_val)
        st.dataframe(general_summary, hide_index=True)
    elif radio_val in cat_class:
        summary_freq = summary_for_categorical(data, stat_col='Status', var_col=radio_val)
        st.table(summary_freq)
