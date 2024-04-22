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

    df = stata_reader.read(preserve_dtypes=False, convert_categoricals=True, convert_dates=True)

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

def zone_of_origin(X:pd.DataFrame)->pd.DataFrame:

    recode_dict = {
            "Andhra Pradesh"             :"Southern Zone", 
            "Arunachal Pradesh"          :"Eastern Zone",
            "Assam"                      :"Eastern Zone", 
            "Bihar"                      :"Eastern Zone", 
            "Chhattisgarh"               :"Central Zone",
            "Goa"                        :"Southern Zone", 
            "Gujarat"                    :"Western Zone",
            "Haryana"                    :"Northern Zone", 
            "Himachal Pradesh"           :"Northern Zone", 
            "Jammu and Kashmir"          :"Northern Zone", 
            "Jharkhand"                  :"Eastern Zone",
            "Karnataka"                  :"Southern Zone", 
            "Kerala"                     :"Southern Zone", 
            "Madhya Pradesh"             :"Central Zone",
            "Maharashtra"                :"Western Zone", 
            "Manipur"                    :"Eastern Zone", 
            "Meghalaya"                  :"Eastern Zone", 
            "Mizoram"                    :"Eastern Zone", 
            "Nagaland"                   :"Eastern Zone", 
            "Odisha"                     :"Eastern Zone", 
            "Punjab"                     :"Northern Zone", 
            "Rajasthan"                  :"Northern Zone",
            "Sikkim"                     :"Eastern Zone", 
            "Tamil Nadu"                 :"Southern Zone", 
            "Telangana"                  :"Southern Zone", 
            "Tripura"                    :"Eastern Zone", 
            "Uttar Pradesh"              :"Central Zone",
            "Uttarakhand"                :"Central Zone",
            "West Bengal"                :"Eastern Zone",
            "Andaman and Nicobar Islands":"Southern Zone",
            "Chandigarh"                 :"Northern Zone",
            "Dadra and Nagar Haveli"     :"Western Zone", 
            "Daman and Diu"              :"Western Zone", 
            "Delhi"                      :"Northern Zone",
            "Lakshadweep"                :"Southern Zone", 
            "Pondicherry"                :"Southern Zone"
        }

    X['Zone of Origin'] = X['State of Origin'].apply(
        lambda x: recode_dict[x] if x is not None else None
    )
    X['Zone of Origin'] = X["Zone of Origin"].astype("category")

    return X

def education_level(X:pd.DataFrame)->pd.DataFrame:

    def converter(x):

        if x is None: return None 

        if x == 0: return 'Illiterate'
        elif x <= 7: return '1 to 7'
        elif x <= 12: return '8 to 12'
        else:
            return 'Above 12'

    X['Education Level'] = X['Years of Education'].apply(
        lambda x: converter(x)
    )

    X['Education Level'] = X['Education Level'].astype("category")

    return X
