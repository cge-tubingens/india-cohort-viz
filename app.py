import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import os
import sys
import json

from pandas.io.stata import StataReader
from india_cohort_viz.Helpers import data_loader

# get current working directory
path_to_wd = os.getcwd()

# append path to local modules for streamlit use
sys.path.append(
    os.path.join(path_to_wd, 'india_cohort_viz')
)

def main(path_to_data=None):

    # check existence of path
    if path_to_data is None:
        raise FileExistsError("Path to data must be given upon initialization.")
    
    # load and parse stata file
    df = data_loader(path_to_stata_file=path_to_data, cwd=path_to_wd)
    
    st.set_page_config(layout="wide")

    st.title('Clinical and Epidemiological Study of PD in India')

    st.write('Here you can explore and visualize data from the biggest study in India on Parkinson disease')

    path_to_types = os.path.join(
        os.getcwd(), os.path.join('auxiliar_data', 'column_type.JSON')
    )
    with open(path_to_types, 'r') as file:
        column_types = json.load(file)

    continuous = column_types["cont_var"]
    categorical= column_types["cat_exc"]

    # Create two columns for the left and right panels
    col1, col2 = st.columns([0.2, 0.6])
    radio_options = list(df.columns)

    pass
