import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import n_colors

import os
import sys
import json

from india_cohort_viz.Helpers import data_loader, show_data, zone_of_origin, education_level
from india_cohort_viz import Summary

# get current working directory
path_to_wd = os.getcwd()

# append path to local modules for streamlit use
sys.path.append(
    os.path.join(path_to_wd, 'india_cohort_viz')
)

def execute_main(path_to_data=None):

    # check existence of path
    if path_to_data is None:
        raise FileExistsError("Path to data must be given upon initialization.")
    
    # load and parse stata file
    df = data_loader(path_to_stata_file=path_to_data, cwd=path_to_wd)
    
    st.set_page_config(layout="wide")

    st.title('Clinical and Epidemiological Study of PD in India')

    st.write('Here you can explore and visualize data from the biggest study in India on Parkinson disease')

    # load JSON files with column types
    path_to_types = os.path.join(
        os.getcwd(), os.path.join('auxiliar_data', 'column_type.JSON')
    )
    with open(path_to_types, 'r') as file:
        column_types = json.load(file)

    df = zone_of_origin(df)
    df = education_level(df)

    # split columns in continuous and cetegorical features
    continuous = []
    categorical= []

    for key in column_types.keys():
        if column_types[key] == "continuous":
            continuous.append(key)
        else:
            categorical.append(key)

    # Create two columns for the left and right panels
    col1, col2 = st.columns([0.2, 0.6])
    radio_options = list(column_types.keys())

    with col1:
        radio_value = st.radio(
            label='Select an option',
            options=radio_options
        )

    with col2:
        col2.header('Summary')

        if radio_value in continuous:

            col2_1, col2_2 = st.columns(2)

            with col2_1:
                st.text('Descriptive statistics')
                show_data(radio_value, df, continuos=continuous)

                st.text('Some statistical tests and its results')
                tests = Summary.hypothesis_test_for_continuous(df, stat_col='Status', var_col=radio_value)
                st.dataframe(tests, hide_index=True)
            with col2_2:
                fig = go.Figure()
                colors = n_colors(
                    'rgb(5, 200, 200)', 'rgb(200, 10, 10)', 
                    len(df['Status'].unique()), colortype='rgb'
                )

                for status, color in zip(df['Status'].unique(), colors):
                    values = df[df['Status']==status][radio_value].copy()
                    fig.add_trace(
                        go.Violin(
                            x=values, 
                            line_color=color, 
                            points='outliers', 
                            name=f"{status}"
                        )
                    )
                fig.update_traces(orientation='h', side='positive', width=3)
                fig.update_layout(xaxis_showgrid=True, xaxis_zeroline=False)
                fig.update_traces(marker_line_width=1, jitter=0.1, pointpos=-0)
                st.plotly_chart(fig, use_container_width=True, theme=None)

        else:
            col2_1, col2_2 = st.columns(2)

            with col2_1:
                st.text('Absolute values and per cents')
                show_data(radio_value, df, cat_class=categorical)

            with col2_2:
                df_count = Summary.prep_stacked_plot(df, status_col='Status', cat_col=radio_value)
                fig_st = px.bar(
                    df_count, 
                    x='Status', 
                    y=['percent'], 
                    color=radio_value, 
                    text=df_count['count'].apply(lambda x: '{0:1.2f}'.format(x)),
                    template='plotly_dark'
                )
                st.plotly_chart(fig_st, theme=None)

    pass

if __name__=="__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]

    execute_main(file_path)
