import pandas as pd
from pandas.io.stata import StataReader

PATH = '/mnt/0A2AAC152AABFBB7/data/LuxGiantMatched/AGESEXMATCHED_GAPINDIA_DATA_23.01.2024.dta'

stata = StataReader(PATH)
df = stata.read(preserve_dtypes=False, convert_categoricals=False, convert_dates=True)
df
