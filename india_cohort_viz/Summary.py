import pandas as pd
import numpy as np
import scipy.stats as sts
import re

def summary_by_centers(X:pd.DataFrame, id_col:str, status_col:str, center_col:str)->pd.DataFrame:

    df_centers = X[[center_col, status_col, id_col]]\
        .groupby( 
            by=[center_col, status_col],
            as_index=False
        ).count()

    df_pivoted = df_centers\
        .pivot(
            index  =center_col, 
            columns=status_col, 
            values =id_col
        ).reset_index().reset_index(drop=True)
    df_pivoted.columns.name = None
    
    new_cols = []
    for col in df_pivoted.columns:
        if col == center_col: new_cols.append('Centers')
        else: new_cols.append(col)

    df_pivoted.columns = new_cols

    df_pivoted['Total'] = df_pivoted[df_pivoted.columns[1]] + df_pivoted[df_pivoted.columns[2]]
    df_pivoted.loc[18] = df_pivoted[df_pivoted.columns[1:]].sum()
    df_pivoted = df_pivoted.fillna('Total')

    return df_pivoted

def control_frequency(X:pd.DataFrame, id_col:str, status_col:str)->pd.DataFrame:

    X_copy = X[[id_col, status_col]].copy()

    df_group = X_copy[[status_col, id_col]].groupby(by=status_col, as_index=False).count()
    df_group.columns = ['Status', 'Count']
    df_group['PerCent'] = df_group['Count']/X_copy.shape[0]

    return df_group

def descriptive_PD_duration(X:pd.DataFrame, status_col:str, sex_col:str, dur_col:str)->pd.DataFrame:

    X_copy = X[[status_col, sex_col, dur_col]].copy()

    X_copy = X_copy[X_copy[status_col]=='Patient']\
        .reset_index(drop=True)\
        .drop(columns=status_col)
    
    df_group = X_copy.groupby(by=sex_col, as_index=False)\
        .agg(
            Median= ('PD_duration', np.median),
            Q1    = ('PD_duration',lambda x: x.quantile(0.25)),
            Q3    = ('PD_duration',lambda x: x.quantile(0.75)),
            IQR   = ('PD_duration',lambda x: x.quantile(0.75)-x.quantile(0.25))
        )

    return df_group

def descriptive_age(X:pd.DataFrame, ctrl_age:str, onset_age:str, status_col:str)->pd.DataFrame:

    X_copy = X[[ctrl_age, onset_age, status_col]].copy()

    X_copy['age2stats'] = X_copy.apply(
        lambda row: row[0] if row [2]=='Control' else row[1], axis=1
    )

    X_copy = X_copy.drop(columns=[ctrl_age, onset_age], inplace=False)

    df_1 = X_copy.groupby(by=status_col, as_index=False)\
        .agg(
            Mean  = ('age2stats', np.mean),
            SD    = ('age2stats', np.std),
            Median= ('age2stats', np.median),
            Q1    = ('age2stats',lambda x: x.quantile(0.25)),
            Q3    = ('age2stats',lambda x: x.quantile(0.75)),
            IQR   = ('age2stats',lambda x: x.quantile(0.75)-x.quantile(0.25))
        )

    return df_1

def cat_and_status(X:pd.DataFrame, status_col:str, cat_col:str)->pd.DataFrame:

    df_grouped = X.groupby(status_col)[cat_col].value_counts(normalize=True).unstack(cat_col)

    return df_grouped

def risk_factors(X:pd.DataFrame, risk_cols:dict, status_col:dict, code:str)->pd.DataFrame:

    X_risk  = X[risk_cols.keys()].copy()
    X_status= X[status_col.keys()].copy().rename(columns=status_col)

    
    mask = (X_risk==code)
    for col in mask.columns:
        mask[col] = mask[col].astype(int)
    mask.rename(columns=risk_cols, inplace=True)

    df = pd.concat([mask, X_status], axis=1)

    df_grouped = df.groupby(by=[col for col in status_col.values()])[[col for col in risk_cols.values()]].sum()

    return df_grouped.transpose()

def initial_sym(X:pd.DataFrame, sym_cols:dict, status_col:str)->pd.DataFrame:

    import re

    def extract_substring(input_string)->str:
        # Define the pattern using regular expression
        pattern = r'=(.*?)\)'

        # Use re.search to find the first match in the string
        match = re.search(pattern, input_string)

        # Check if a match is found
        if match:
            # Extract the substring between '=' and ')' using group(1)
            substring = match.group(1)
            return substring
        else:
            return None

    def classifier(row, col_labels:dict)->str:

        col_names = list(col_labels.keys())

        if row[col_names[0]]=='Checked' and row[col_names[1]]=='Unchecked':
            value = extract_substring(col_labels[col_names[0]])
            return value
        elif row[col_names[0]]=='Unchecked' and row[col_names[1]]=='Checked':
            value = extract_substring(col_labels[col_names[1]])
            return value
        elif row[col_names[0]]=='Checked' and row[col_names[1]]=='Checked':
            value = '{} and {}'.format(
                extract_substring(col_labels[col_names[0]]),
                extract_substring(col_labels[col_names[1]])
            )
            return value
        else:
            return None

    cols = list(sym_cols.keys())
    X_copy = X[cols + [status_col]].copy()
    X_copy = X[X[status_col]=='Patient']\
        .reset_index(drop=True)\
        .drop(columns=status_col, inplace=False)
    

    X_copy['Symptoms'] = X_copy.apply(
        lambda row: classifier(row, sym_cols), axis=1
    )

    df_prop = np.round(X_copy['Symptoms'].value_counts(normalize=True)*100,2).reset_index()
    df_prop.columns = ['Initial Symptoms', 'Percentage']
    df_freq = X_copy['Symptoms'].value_counts().reset_index()
    df_freq.columns = ['Initial Symptoms', 'Frequency']

    df = df_prop.merge(df_freq, on='Initial Symptoms')

    return df, np.round(X_copy['Symptoms'].value_counts(normalize=True)*100,2)

def brad_trem_rig(X:pd.DataFrame, symp_cols:dict, status_col:str)->pd.DataFrame:

    X_copy = X[list(symp_cols.keys()) + [status_col]].copy()
    X_copy = X_copy[X_copy[status_col]=='Patient']\
        .reset_index(drop=True).drop(columns=status_col)
    
    new_cols = {}
    for key in symp_cols.keys():
        new_cols[key] = symp_cols[key].split('-')[0].strip()

    X_copy = X_copy.rename(columns=new_cols)

    df = pd.DataFrame({'Side':['Left', 'Right', 'No predominant side']})

    for col in X_copy.columns:
        temp = np.round(X_copy[col].value_counts(normalize=True).sort_index()*100,2).reset_index()
        temp.columns = ['Side', col]
        df = df.merge(temp, on='Side')

    return df.set_index('Side', inplace=False).transpose()

def motor_symptoms(X:pd.DataFrame, motor_cols:dict, status_col)->pd.DataFrame:

    import re

    def extract_substring(input_string):
        # Define the pattern using regular expression
        pattern = r'=(.*?)\)'

        # Use re.search to find the first match in the string
        match = re.search(pattern, input_string)

        # Check if a match is found
        if match:
            # Extract the substring between '=' and ')' using group(1)
            substring = match.group(1)
            return substring
        else:
            return None

    X_copy = X[list(motor_cols.keys()) + [status_col]].copy()
    X_copy = X_copy[X_copy[status_col]=='Patient']\
        .reset_index(drop=True).drop(columns=status_col)
    
    new_cols = {}
    for key in motor_cols.keys():
        new_cols[key] = extract_substring(motor_cols[key])

    X_copy = X_copy.rename(columns=new_cols)
    
    mask = (X_copy=='Checked')
    df = np.round(mask.sum()/X_copy.shape[0]*100,2)
    
    return df

def dbs_summary(X:pd.DataFrame, dbs_col:str, status_col:str, num_columns:list, cat_colums:list=[])->pd.DataFrame:

    def Q1(g:pd.Series):
        return g.quantile(0.25)
    def Q3(g:pd.Series):
        return g.quantile(0.75)
    
    agg_funcs = {}
    if len(num_columns)>1:
        for col in num_columns:
                agg_funcs[col] = [np.median, Q1, Q3]
    if len(cat_colums)>1:
        for col in cat_colums:
                agg_funcs[col] = ['count']

    cols = [dbs_col, status_col]
    X_copy = X[cols + num_columns + cat_colums]\
        .copy().dropna(subset=dbs_col, inplace=False)

    X_copy = X_copy[X_copy[status_col]=='Patient'].reset_index(drop=True).drop(columns=status_col)

    df_pivoted = pd.pivot_table(
        X_copy, 
        values=num_columns, 
        columns=dbs_col, 
        aggfunc=agg_funcs
    )

    pivoted_dict = {}
    pivoted_dict['Feature'] = list(df_pivoted.index.levels[0])
    for cat in df_pivoted.columns.categories:
        key = cat+' DBS'
        pivoted_dict[key] = []
        for level_1 in df_pivoted.index.levels[0]:
            pivoted_dict[key].append('{}({}-{})'.format(df_pivoted.loc[(level_1, 'median'),cat], 
                                           df_pivoted.loc[(level_1, 'Q1'),cat],
                                           df_pivoted.loc[(level_1, 'Q3'),cat]))
    df = pd.DataFrame(pivoted_dict)
    
    total_dict = {}
    total_dict['Feature'] = num_columns
    total_dict['Total'] = []
    for col in num_columns:
        total_dict['Total'].append(
            '{}({}-{})'.format(X_copy[col].median(),
                               X_copy[col].quantile(0.25),
                               X_copy[col].quantile(0.75))
        )
    total = pd.DataFrame(total_dict)
    df = df.merge(total, on='Feature')

    df_temp = pd.DataFrame()
    for col in cat_colums:
        df_1 = X_copy[col].value_counts().reset_index()
        df_1.columns = ['Feature', 'count']
        df_temp = pd.concat([df_temp, df_1], axis=1)

    cat_dict = {}
    cat_dict['Feature'] = df_temp['Feature']
    cat_dict['Total'] = df_temp['count'].apply(lambda x: '{}(100)'.format(x))
    df_cats_total = pd.DataFrame(cat_dict)

    df_cats_partial = pd.DataFrame()

    for col in cat_colums:
        df_freq = X_copy[[dbs_col, col]].groupby(by=[dbs_col, col], as_index=False)\
            .size()\
            .pivot(values='size', columns=dbs_col, index=col)\
            .reset_index()
        df_ratio = df_freq.copy()
        denom = df_freq['Yes'] + df_freq['No']
        df_ratio['Yes']= np.round(100*df_ratio['Yes']/denom,2)
        df_ratio['No'] = np.round(100*df_ratio['No']/denom,2)

        new_dict = {
            'Feature': df_freq[col]
        }
        for cat in df_pivoted.columns.categories:
            key = cat+' DBS'
            new_dict[key] = ['{} ({})'.format(df_freq.loc[k,cat], df_ratio.loc[k,cat]) for k in range(df_freq.shape[0])]
        df_cats_partial = pd.concat([df_cats_partial, pd.DataFrame(new_dict)])

    df_cats = pd.merge(df_cats_partial, df_cats_total, on='Feature')

    return pd.concat([df, df_cats], axis=0, ignore_index=True)

def compact_quantiles(X:pd.DataFrame, variable: str, category:str)->pd.DataFrame:

    values = X.apply(
        lambda row: '{} ({}-{})'.format(row['Median'], row['Q1'], row['Q3']), axis=1
    )

    return pd.DataFrame({category: X[X.columns[0]],
                         '{}\n [Median (IQR)]'.format(variable):values})

def hoehn_and_yahr_summary(X:pd.DataFrame, cols:list)->pd.DataFrame:

    X_copy = X[cols].copy()

    X_copy = X_copy[X_copy[cols[0]]=='Patient'].reset_index(drop=True)
    X_copy = X_copy.drop(columns=cols[0], inplace=False)
    
    temp_dur = X_copy.groupby(by=cols[1], as_index=False)\
        .agg(
            Median= (cols[2], np.median),
            Q1    = (cols[2], lambda x: np.round(x.quantile(0.25), 2)),
            Q3    = (cols[2], lambda x: np.round(x.quantile(0.75), 2)),
            IQR   = (cols[2], lambda x: np.round(x.quantile(0.75)-x.quantile(0.25), 2))
        )
    temp_age = X_copy.groupby(by=cols[1], as_index=False)\
        .agg(
            Median= (cols[3], np.median),
            Q1    = (cols[3], lambda x: np.round(x.quantile(0.25), 2)),
            Q3    = (cols[3], lambda x: np.round(x.quantile(0.75), 2)),
            IQR   = (cols[3], lambda x: np.round(x.quantile(0.75)-x.quantile(0.25), 2))
        )
    
    df_dur = compact_quantiles(temp_dur, 'Disease Duration', category='Hoeh and Yahr Staging')
    df_age = compact_quantiles(temp_age, 'Age at Onset', category='Hoeh and Yahr Staging')
    
    return pd.merge(df_dur, df_age, on='Hoeh and Yahr Staging')


def symptons_prevalence_computation(X:pd.DataFrame, symptoms:dict, status:str)->pd.DataFrame:

    """
    Function to compute the prevalence of certain symptoms among a population of Patients.

    Parameters
    ----------

    X: pandas.DataFrame
        DataFrame with the clinical information of all the population.
    symptoms: dict
        dictionary where the keys are the names of the columns in the dataframe and its values is the corresponding symptom.
    status: str
        column name where it is stored the status of the observation (Patient or Control)

    Return: pandas.DataFrame
    ------------------------
    Data frame with the percent of each symptom in the patients population.
    """

    X_copy = X[[status] + list(symptoms.keys())].copy()
    X_copy = X_copy[X_copy[status]=='Patient'].reset_index(drop=True)\
        .drop(columns=status, inplace=False)

    boolean_df = (X_copy[list(symptoms.keys())] == 'Checked')
    df_prev = boolean_df.sum()/boolean_df.shape[0]
    df_prev = 100*df_prev
    df_prev = df_prev.reset_index()
    df_prev.columns = ['Symptom', 'Prevalence']
    df_prev = df_prev.sort_values(by='Prevalence', ascending=False).reset_index(drop=True)
    
    return df_prev

def extract_substring(input_string):
        # Define the pattern using regular expression
        pattern = r'=(.*?)\)'

        # Use re.search to find the first match in the string
        match = re.search(pattern, input_string)

        # Check if a match is found
        if match:
            # Extract the substring between '=' and ')' using group(1)
            substring = match.group(1)
            return substring
        else:
            return None

def symptoms_age_pd_duration(X:pd.DataFrame, cols:list, symptoms:dict)->pd.DataFrame:

    X_copy = X[cols + list(symptoms.keys())].copy()
    X_copy = X_copy[X_copy[cols[0]]=='Patient'].reset_index(drop=True)
    X_copy = X_copy.dropna(subset=cols[1])

    df_prev = symptons_prevalence_computation(X_copy, symptoms, cols[0])

    X_copy = X_copy.drop(columns=cols[0], inplace=False)
    X_copy['Disease Duration Cut-off'] = X_copy[cols[1]].apply(
        lambda x: '<=5 years' if x<=5 else '>5 years'
    )

    df = pd.DataFrame(columns=['Symptom', 'Disease Dur. <=5 years', 'Disease Dur. >5 years'])

    for key in symptoms.keys():
        temp = X_copy[['Disease Duration Cut-off', key, cols[2]]].copy()
        temp = temp[temp[key]=='Checked'].reset_index(drop=True)

        temp_grouped = temp.groupby(by='Disease Duration Cut-off')\
            .agg(
                Median= (cols[2], np.median),
                Q1    = (cols[2], lambda x: np.round(x.quantile(0.25), 2)),
                Q3    = (cols[2], lambda x: np.round(x.quantile(0.75), 2)),
            )
        
        val_less_5 = '{} ({}-{})'.format(
            temp_grouped.loc['<=5 years', 'Median'],
            temp_grouped.loc['<=5 years', 'Q1'],
            temp_grouped.loc['<=5 years', 'Q3']
        )

        val_more_5 = '{} ({}-{})'.format(
            temp_grouped.loc['>5 years', 'Median'],
            temp_grouped.loc['>5 years', 'Q1'],
            temp_grouped.loc['>5 years', 'Q3']
        )

        new_row = {
            'Symptom': key,
            'Disease Dur. <=5 years': val_less_5,
            'Disease Dur. >5 years':val_more_5
        }

        df.loc[len(df)] = new_row

    df = pd.merge(df_prev, df, on='Symptom')
    df['Symptom'] = df['Symptom'].apply(lambda x: extract_substring(symptoms[x]))
        

    return df

def fill_age_at_onset(status:str, age_assess:float, age_onset:float)->pd.Series:

    if status == 'Patient':
        return age_onset
    else:
        return age_assess
    
def f_test(x, y):
    """
    https://www.statology.org/f-test-python/
    """
    
    f = np.nanvar(x, ddof=1)/np.nanvar(y, ddof=1) #calculate F test statistic
    dfn = x.size-1 #define degrees of freedom numerator 
    dfd = y.size-1 #define degrees of freedom denominator 
    
    p = 1-sts.f.cdf(f, dfn, dfd) #find p-value of F test statistic 
    
    return f, p
    
def continuous_variable_summary(X:pd.DataFrame, continuous_feat:list)->pd.DataFrame:

    X_copy = X[continuous_feat].copy()

    stats1 = ['N', 'Overall', 'Patients', 'Controls', 'F-test p-value', 'Ansari-Bradley p-value',
              'K-S p-value (Control)', 'K-S p-value (Patient)', 'Mann-Whitney p-value']
    stats2 = ['N', 'Overall', 'Patients', 'Controls']

    df_summary1 = pd.DataFrame(columns=stats1, index=continuous_feat)
    df_summary2 = pd.DataFrame(columns=stats2, index=continuous_feat)

    for col in continuous_feat:
        df_summary1.loc[col] = {
            'N': np.round((~X_copy[col].isnull()).sum(),0),
            'Overall': '{} ({})'.format(np.round(X_copy[col].mean(),2), np.round(X_copy[col].std(),2))
        }

        df_summary2.loc[col] = {
            'N': np.round((~X_copy[col].isnull()).sum(),0),
            'Overall': '{} ({}-{})'.format(
                np.round(X_copy[col].mean(),2), 
                np.round(np.nanpercentile(X_copy[col], 25),2),
                np.round(np.nanpercentile(X_copy[col], 75),2)
            )
        }

    for col in continuous_feat:
        temp = X[['Status', col]].groupby(by='Status').agg(
            Median= (col, np.median),
            Q1    = (col, lambda x: np.round(x.quantile(0.25), 2)),
            Q3    = (col, lambda x: np.round(x.quantile(0.75), 2))
    )
        df_summary2.loc[col, 'Controls'] = '{} ({}-{})'.format(
            np.round(temp.loc['Control', 'Median'],2), temp.loc['Control', 'Q1'], temp.loc['Control', 'Q3']
        )
        df_summary2.loc[col, 'Patients'] = '{} ({}-{})'.format(
            np.round(temp.loc['Patient', 'Median'],2), temp.loc['Patient', 'Q1'], temp.loc['Patient', 'Q3']
        )

    for col in continuous_feat:
        temp = X[['Status', col]].groupby(by='Status').agg(
            Mean  = (col, np.mean),
            Std   = (col, np.std)
    )
        df_summary1.loc[col, 'Controls'] = '{} ({})'.format(
            np.round(temp.loc['Control', 'Mean'],2), np.round(temp.loc['Control', 'Std'],2)
        )
        df_summary1.loc[col, 'Patients'] = '{} ({})'.format(
            np.round(temp.loc['Patient', 'Mean'],2), np.round(temp.loc['Patient', 'Std'],2)
        )

    for col in continuous_feat:
        temp_x = X[X['Status']=='Patient'].reset_index(drop=True)
        temp_y = X[X['Status']=='Control'].reset_index(drop=True)

        x = temp_x[col].copy().to_numpy()
        y = temp_y[col].copy().to_numpy()

        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]

        f,p = f_test(x,y)

        df_summary1.loc[col, 'F-test p-value'] = p
        df_summary1.loc[col, 'Ansari-Bradley p-value'] = sts.ansari(x,y)[1]
        df_summary1.loc[col, 'K-S p-value (Control)'] = sts.kstest(y, 'norm')[1]
        df_summary1.loc[col, 'K-S p-value (Patient)'] = sts.kstest(x, 'norm')[1]
        df_summary1.loc[col, 'Mann-Whitney p-value'] = sts.mannwhitneyu(x, y, nan_policy='omit')[1]

    return df_summary1, df_summary2

def summary_for_continuos(X:pd.DataFrame, stat_col:str, var_col:str)->pd.DataFrame:

    new_index = ['N', 'Mean', 'Std Dev', 'Min', 'Q1', 'Median', 'Q3', 'Max']

    overall = X[var_col].describe().reset_index()
    overall.columns = [var_col, 'Overall']

    temp_pat = X[X[stat_col]=='Patient'].reset_index(drop=True)
    temp_con = X[X[stat_col]=='Control'].reset_index(drop=True)
    

    patient = temp_pat[var_col].describe().reset_index()
    patient.columns = [var_col, 'Patients']
    control = temp_con[var_col].describe().reset_index()
    control.columns = [var_col, 'Controls']

    df = overall.merge(patient, on=var_col).merge(control, on=var_col)
    df[var_col] = new_index

    return df

def hypothesis_test_for_continuous(X:pd.DataFrame, stat_col:str, var_col:str)->pd.DataFrame:
    
    temp_pat = X[X[stat_col]=='Patient'].reset_index(drop=True)
    patient = temp_pat[var_col].to_numpy()
    patient = patient[~np.isnan(patient)]

    temp_con = X[X[stat_col]=='Control'].reset_index(drop=True)
    control = temp_con[var_col].to_numpy()
    control = control[~np.isnan(control)]

    test_names = ['F-test', 'Ansari-Bradley', 'Kolmogorov-Smirnov (Control)', 
                  'Kolmogorov-Smirnov (Patient)', 'Mann-Whitney']
    p_values   = []
    test_values= []
    
    tests = [
        f_test(patient, control),  sts.ansari(patient, control), sts.kstest(control, 'norm'),
        sts.kstest(patient, 'norm'), sts.mannwhitneyu(patient, control, nan_policy='omit')
    ]
    for tup in tests:
        p_values.append(tup[1])
        test_values.append(tup[0])

    return pd.DataFrame({
        'Test name': test_names, 'p-value': p_values, 'Test value': test_values
    })

def summary_for_categorical(X:pd.DataFrame, stat_col:str, var_col:str)->pd.DataFrame:


    overall_freq = X[var_col].value_counts().to_frame()
    overall_freq.columns = ['Overall freq']

    overall_rat = X[var_col].value_counts(normalize=True)*100
    overall_rat = overall_rat.to_frame()
    overall_rat.columns = ['Overall %']

    temp_pat = X[X[stat_col]=='Patient'].reset_index(drop=True)
    temp_con = X[X[stat_col]=='Control'].reset_index(drop=True)
    

    patient_freq = temp_pat[var_col].value_counts().to_frame()
    patient_freq.columns = ['Patients freq']
    control_freq = temp_con[var_col].value_counts().to_frame()
    control_freq.columns = ['Controls freq']

    patient_rat = temp_pat[var_col].value_counts(normalize=True)*100
    patient_rat = patient_rat.to_frame()
    patient_rat.columns = ['Patients %']
    control_rat = temp_con[var_col].value_counts(normalize=True)*100
    control_rat = control_rat.to_frame()
    control_rat.columns = ['Controls %']

    df_freq = overall_freq.merge(patient_freq, on=var_col).merge(control_freq, on=var_col)
    df_rat = overall_rat.merge(patient_rat, on=var_col).merge(control_rat, on=var_col)

    df = pd.merge(df_freq, df_rat, on=var_col)

    multi_index = pd.MultiIndex.from_product(
        [['Overall', 'Patients', 'Controls'], ['freq', '%']],
        sortorder=0, names=['Group', 'Statistic']
    )
    DF = pd.DataFrame(index=df.index.categories, columns=multi_index)
    
    for row in df.index:
        for multi in multi_index:
            col = multi[0] + ' ' + multi[1]
            DF.loc[row, multi] = df.loc[row, col]

    return DF

def prep_stacked_plot(df, status_col:str, cat_col:str):

    cols = [status_col, cat_col]
    X_copy = df[cols].copy()
    X_copy['count'] = 1

    df_count = X_copy.groupby(by=cols).count()
    counter = pd.Series(data=df_count['count'], index=df_count.index, name='Count')

    df_count['percent'] = counter.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values

    df_count = df_count.reset_index()

    return df_count
