####
# This program creates a classifier for ferret data using PCA and logisitic regression. It is designed to discriminate
# between uninjured (control) and injured (vehicle) animals. This model is then applied to animals that have been
# treated with Epo and hypothermia (TH). This program uses a datafiles that includes the multiple runs for each animal.
####

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from helper_functions import remove_columns_with_substring, write_list_to_file, plot_feature_distributions
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from ShuffleSeries import shuffle_series
from Run_log_reg_on_ttsplit_with_treat import run_log_reg_on_ttsplit

# Choose target variable
target = 'total gross score' #options: 'combined injury score', 'Pathology Score', 'total gross score', 'avg_5.30', 'Overall Sulci Sum', 'Overall Gyri Sum'
# Choose group
group_to_test = 'Control and Vehicle'
outputdir = '/home/toddr/neva/PycharmProjects/WoodAnimalData'

# Set number of splits for stratified k fold cross validation
n_splits = 5

#Load ferret data with injury information
injury = pd.read_csv('CW_GI_Path2.csv')

#Load ferret data with multiple run information
ferret_mrun = pd.read_csv('Ferret CatWalk Run Statistics MulitpleRuns March2024.csv')

#get unique ID values for these dataframes
unique_id_injury_file = injury['ID'].unique().tolist()
unique_id_mrun_file = ferret_mrun['ID'].unique().tolist()

injury_data_no_mrun = [a for a in unique_id_injury_file if a not in unique_id_mrun_file]
mrun_data_no_injury = [a for a in unique_id_mrun_file if a not in unique_id_injury_file]

#copy injury measures from brain from injury datafile to multiple ferret run data file

ferret = pd.merge(ferret_mrun, injury[['ID', 'Pathology Score', 'total gross score',
                                       'avg_5.30', 'Overall Sulci Sum', 'Overall Gyri Sum']], on='ID', how='left')

# Select rows containing the string '-'
rows_with_dash = ferret[ferret.apply(lambda row: '-' in row.values, axis=1)]
# Write these rows to a new dataframe
dash_df = pd.DataFrame(rows_with_dash)

#count number of rows that contain string '-'
count = (ferret == '-').any(axis=1).sum()
print("Number of rows containing the string '-':", count)
#replace '-' with nan
ferret.replace('-', np.nan, inplace=True)

## Count total number of null values for each column
pd.set_option('display.max_rows' , None)
print(" \nCount total null values for each column  : \n\n",ferret.isnull().sum())

#remove rows with nans
ferret.dropna(inplace=True)
ferret.reset_index(inplace=True, drop=True)

## Count total number of null values in dataframe after nan removal
pd.set_option('display.max_rows' , None)
print(" \nAfter nan removal, count total null values in dataframe : \n\n",ferret.isnull().sum().sum())

#recode gender as binary feature M=1, F=0
ferret['Sex'] = ferret['Sex'].replace('F', 0) #female = 0
ferret['Sex'] = ferret['Sex'].replace('M', 1) #male = 1

#make total gross score a binary columns because there are so few cases with values >1
ferret.loc[ferret['total gross score'] > 1, 'total gross score'] = 1

# give every subject group a number
controlcode = 0  # Control = 0
vehiclecode = 1  # Vehicle = 1
epocode = 2  # Epo = 2
thcode = 3  # TH = 3

ferret['Group'] = ferret['Group'].replace(['Control'], controlcode)
ferret['Group'] = ferret['Group'].replace(['Veh'], vehiclecode)
ferret['Group'] = ferret['Group'].replace(['Vehicle'], vehiclecode)
ferret['Group'] = ferret['Group'].replace(['Epo'], epocode)
ferret['Group'] = ferret['Group'].replace(['TH'], thcode)

#Plot histograms of features
# plot_feature_distributions(ferret, ferret.columns)

#calculate correlation matrix
print(ferret.dtypes)

#the columns LF and RF_MinIntensityMean has a constant value of 8 for all rows so drop this column
ferret.drop(columns=['RF_MinIntensity_Mean', 'LF_MinIntensity_Mean'], inplace=True)
#Ask about StepSequence_RB_(%), StepSequence_RA_(%), Support_Zero_(%)

group_index = epocode

group_of_interest_indices = ferret.index[ferret['Trx'] < group_index].tolist()

ferret = ferret.iloc[group_of_interest_indices, :]
ferret.reset_index(inplace=True, drop=True)

#select feature columns
all_columns=ferret.columns.tolist()
subj_info_columns = ['ID', 'Trx']

response_columns = ['Pathology Score', 'total gross score', 'avg_5.30','Overall Sulci Sum', 'Overall Gyri Sum']
feature_columns = [x for x in all_columns if ((x not in subj_info_columns) and (x not in response_columns))]

# Split the data into 80/20 train-test split stratified by the target column in the X dataframe
stratified_split = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

nshuffles=1000
unique_shuffled_series = shuffle_series(ferret[target], nshuffles)

ferret_orig=ferret.copy()

# create column transformer for standard scalar
# choose all columns except for the first column (sex)
columns_to_scale = list(range(1, len(feature_columns)))
transformer = ColumnTransformer(transformers=[('scale', StandardScaler(), columns_to_scale)],
                                remainder='passthrough')

# Create the pipeline for logistic regression (use for Pathology Score and total gross score which are made binary above)
pipe_logreg = Pipeline([
    ('t', transformer),
    ('pca', PCA(n_components=21)),
    ('logreg', LogisticRegression(solver='liblinear', penalty='l1', C=0.04, random_state=42))
])

auctrain_list=[]
auctest_list=[]
for rep in range(nshuffles):
    print(f'REP {rep}')

    shuff = pd.Series(unique_shuffled_series[rep])
    ferret[target] = shuff

    #Make a dataframe with just features and a target dataframe with just the response variable
    X = ferret[feature_columns].copy()
    y = ferret[[target]].copy()

    final_avg_auc_roc_train, final_avg_auc_roc_test = run_log_reg_on_ttsplit(stratified_split, X, y, target,n_splits, pipe_logreg)

    auctrain_list.append(final_avg_auc_roc_train)
    auctest_list.append(final_avg_auc_roc_test)

    mystop=1

write_list_to_file(auctrain_list, f'{outputdir}/train_auc_roc_shuffle_list.csv')
write_list_to_file(auctest_list, f'{outputdir}/test_auc_roc_shuffle_list.csv')


mystop=1