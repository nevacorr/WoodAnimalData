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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.compose import ColumnTransformer
from ShuffleSeries import shuffle_series
from Run_log_reg_on_ttsplit_with_treat import run_log_reg_on_ttsplit
import sys

# Choose target variable
target = 'total gross score' #options: 'Pathology Score', 'total gross score', 'avg_5.30', 'Overall Sulci Sum', 'Overall Gyri Sum'
# Choose group
group_to_test = 'Control and Vehicle'
outputdir = '/home/toddr/neva/PycharmProjects/WoodAnimalData'

# Set number of splits for stratified k fold cross validation
n_splits = 5

# Load ferret data with multiple run information
ferret = pd.read_csv('data/Ferret CatWalk EpoTH IDs 60-74 Run Statistics with Brain Morphology.csv')

ferret_orig=ferret.copy()

# Brain regional size measurements
columns_brain_volumes = ['total volume (cm^3)', 'cerebrum+brainstem (cm^3)', 'cerebellum (cm^3)', '% cerebellum', 'Summed White Matter GFAP (um)',
                     'CC Thickness (um)', 'Overall Sulci Sum', 'Overall Gyri Sum']

# Remove brain volume columns
ferret.drop(columns=columns_brain_volumes, inplace=True)

# Remove rows with trial and run information and walkway width
ferret.drop(columns=['Trial', 'Run', 'WalkWay_Width_(cm)'], inplace=True)

# Remove rows with constant values
ferret = ferret.loc[:, (ferret != ferret.iloc[0]).any()]

# Find missing data. Missing values are indicated with nan or '-'. Select rows containing the string '-'
rows_with_dash = ferret[ferret.apply(lambda row: '-' in row.values, axis=1)]
# Write these rows to a new dataframe
dash_df = pd.DataFrame(rows_with_dash)

#Find missing data count number of rows that contain string '-'
count = (ferret == '-').any(axis=1).sum()
print("Number of rows containing the string '-':", count)
#replace '-' with nan
ferret.replace('-', np.nan, inplace=True)

# ## Count and print total number of null values for each column with nans
nan_cols = ferret.isna().sum()
columns_with_nans = nan_cols[nan_cols > 0]
print(columns_with_nans)

#remove rows with nans
ferret.dropna(inplace=True)
ferret.reset_index(inplace=True, drop=True)

# Recode gender as numeric categorical feature M=1, F=0
ferret['Sex'] = ferret['Sex'].replace('F', 0) #female = 0
ferret['Sex'] = ferret['Sex'].replace('M', 1) #male = 1

# Give target variables simpler names
ferret.rename(columns={'Cortical Lesion (0-4) + Mineralization (0-4)': 'total gross score',
               'Morph. Injury Score (path+BM+WM)': 'Pathology Score'}, inplace=True)

# Make total gross score a binary columns because there are so few cases with values >1
ferret.loc[ferret['total gross score'] > 1, 'total gross score'] = 1

# give every subject group a number
controlcode = 0  # Control = 0
vehiclecode = 1  # Vehicle = 1
epocode = 2  # Epo = 2
thcode = 3  # TH = 3

# Code group as number instead of string
ferret['Group'] = ferret['Group'].replace(['Control'], controlcode)
ferret['Group'] = ferret['Group'].replace(['Veh'], vehiclecode)
ferret['Group'] = ferret['Group'].replace(['Vehicle'], vehiclecode)
ferret['Group'] = ferret['Group'].replace(['Epo'], epocode)
ferret['Group'] = ferret['Group'].replace(['TH'], thcode)

# Convert any column of type object to type float
for col in ferret.columns:
    if ferret[col].dtype == 'O':
        ferret[col] = pd.to_numeric(ferret[col])

#Plot histograms of features
# plot_feature_distributions(ferret, ferret.columns)

# Check column datatypes
dt = ferret.dtypes

# Remove all cases where group is not control or vehicle
ferret_cv = ferret[ferret['Group'].isin([1,2])]
ferret_cv.reset_index(inplace=True, drop=True)

#select feature columns
all_columns=ferret_cv.columns.tolist()
subj_info_columns = ['ID', 'Group']

response_columns = ['Pathology Score', 'total gross score']
feature_columns = [x for x in all_columns if ((x not in subj_info_columns) and (x not in response_columns))]

# Split the data into 80/20 train-test split stratified by the target column

split_df = ferret_cv.groupby('ID').mean()
split_df = split_df.iloc[:,0:4]
split_df.reset_index(inplace=True)

# Check that the groupby mean operation did not change the values for each subject for the variables with only two possible values
if not split_df['Group'].isin([1.0, 2.0]).all():
    print(f'Error: Values in "Group" column are not limited to 1 and 2. Exiting program')
    sys.exit(1)
if not split_df['Sex'].isin([0.0, 1.0]).all():
    print(f'Error: Values in "Sex" column are not limited to 0 and 1. Exiting program')
    sys.exit(1)
if not split_df['total gross score'].isin([0.0, 1.0]).all():
    print(f'Error: Values in "total gross score" column are not limited to 0 and 1. Exiting program')
    sys.exit(1)

X = split_df[['ID', 'Sex']]
y = pd.DataFrame(split_df[target], columns=[target])

stratify_by = pd.concat([X['Sex'], y[target]], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=stratify_by, test_size=0.2, random_state=42)
Xtrain = ferret_cv[ferret_cv['ID'].isin(X_train['ID'])]
test = ferret_cv[ferret_cv['ID'].isin(X_test['ID'])]

# create column transformer for standard scalar
# choose all columns except for the first column (sex)
columns_to_scale = list(range(1, len(feature_columns)))
transformer = ColumnTransformer(transformers=[('scale', StandardScaler(), columns_to_scale)],
                                remainder='passthrough')

# Create the pipeline for logistic regression (use for total gross score which was made binary)
pipe_logreg = Pipeline([
    ('t', transformer),
    ('pca', PCA()),
    ('logreg', LogisticRegression(solver='liblinear', penalty='l1', C=0.04, random_state=42))
])



mystop=1