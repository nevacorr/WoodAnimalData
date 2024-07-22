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
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
import sys
from matplotlib import pyplot as plt
import seaborn as sns
from data_cleaning_and_exploration import clean_data, calc_corr, remove_highly_corr_features
from data_cleaning_and_exploration import calculate_roc_auc_and_plot, plot_confusion_matrix
from data_cleaning_and_exploration import return_most_important_features
from collections import Counter

# Choose target variable
target = 'total gross score' #options: 'Pathology Score', 'total gross score', 'avg_5.30', 'Overall Sulci Sum', 'Overall Gyri Sum'
# Define project directory location
outputdir = '/home/toddr/neva/PycharmProjects/WoodAnimalData'
num_tt_splits = 10
show_confusion_matrices = 0
show_roc_curve = 0
plot_distributions = 0

# Load ferret data with multiple run information
ferret_orig = pd.read_csv('data/Ferret CatWalk EpoTH IDs 60-74 Run Statistics with Brain Morphology.csv')

# Clean data
ferret = clean_data(ferret_orig)

# Class coding: 0: Control, 1: Vehicle, 2: Epo, 3: TH

if plot_distributions:
    # Plot distributions of features
    plot_feature_distributions(ferret, ferret.columns)

# Remove all cases where group is not control or vehicle
ferret_cv = ferret[ferret['Group'].isin([0,1])]
ferret_cv.reset_index(inplace=True, drop=True)

# Define feature columns
all_columns=ferret_cv.columns.tolist()
subj_info_columns = ['ID', 'Group']
response_columns = ['Pathology Score', 'total gross score']
feature_columns = [x for x in all_columns if ((x not in subj_info_columns) and (x not in response_columns))]

# Calculate lower triangle correlation matrix
ferret_to_corr = ferret_cv[feature_columns]
ferret_to_corr = ferret_to_corr.drop(columns=['Sex'])
lower_tri_corr = calc_corr(ferret_to_corr, plotheatmap=0)

# Make a dataframe of unique subjects with ID and Sex
split_df = ferret_cv.groupby('ID').mean()
split_df = split_df.loc[:,['Sex', target]]
split_df.reset_index(inplace=True)

# Check that the groupby mean operation did not change the values for Sex from 0 and 1
if not split_df['Sex'].isin([0.0, 1.0]).all():
    print(f'Error: Values in "Sex" column are not limited to 0 and 1. Exiting program')
    sys.exit(1)

X = split_df[['ID', 'Sex']].copy()
y = split_df[['ID', target]].copy()

stratify_by = pd.concat([X['Sex'], y[target]], axis=1)

############################## BUILD PIPELINE  ################################

# create column transformer for standard scalar
# choose all columns except for the first column (sex)
columns_to_scale = feature_columns.copy()
columns_to_scale.remove('Sex')
transformer = ColumnTransformer(transformers=[('scale', StandardScaler(), columns_to_scale)],
                                remainder='passthrough')

# Create the pipeline for logistic regression (use for total gross score which was made binary)
pipe_logreg = Pipeline([
    ('t', transformer),
    ('pca', PCA()),
    ('logreg', LogisticRegression(solver='liblinear'))
])

# Define parameter grid
param_grid = {
    'pca__n_components': range(5, 25, 5),  # 25 or more components often leads to overfitting of the data (auc_roc >0.9 for train set)
    'logreg__penalty': ['l1', 'l2'],
    'logreg__C': np.logspace(-3, 1, 5)
}

grid_search = GridSearchCV(pipe_logreg, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

# Initialize summary variables
important_feature_dict = {}
roc_auc_train = []
roc_auc_test = []
best_parameters_list = []

# Perform train test split iteratively with different random_state every time
for i in range(num_tt_splits):

    # Perform train test split of unique subjects
    X_train_groupby, X_test_groupby, y_train_groupby, y_test_groupby = train_test_split(X, y, stratify=stratify_by, test_size=0.2, random_state=None)

    # Assign all data from subjects to train or test X and y dataframes
    X_train = ferret_cv.loc[ferret_cv['ID'].isin(X_train_groupby['ID']), feature_columns]
    X_test = ferret_cv.loc[ferret_cv['ID'].isin(X_test_groupby['ID']), feature_columns]
    y_train = ferret_cv.loc[ferret_cv['ID'].isin(y_train_groupby['ID']), ['ID'] + [target]]
    y_test = ferret_cv.loc[ferret_cv['ID'].isin(y_test_groupby['ID']), ['ID'] + [target]]

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # Extract ID numbers for all rows of train and test sets
    train_IDs = y_train['ID']
    test_IDs = y_test['ID']

    # Drop ID numbers from target dataframe
    y_train.drop(columns=['ID'], inplace=True)
    y_test.drop(columns=['ID'], inplace=True)

    ########################### TRAIN AND FIT LOGISTIC REGRESSION #############################

    # Fit the model to the training data
    grid_search.fit(X_train, y_train.values.ravel())

    # Get the best parameters and best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    best_parameters_list.append(tuple(best_params.items()))

    ########################## MAKE CLASS PREDICTIONS AND EVALUATE PERFORMANCE ########################

    # predict values for the train and test data
    y_hat_train = best_model.predict(X_train)
    y_hat_test = best_model.predict(X_test)
    y_proba_train = best_model.predict_proba(X_train)
    y_proba_test = best_model.predict_proba(X_test)

    # Assign ID numbers to predictions
    y_hat_train_df = pd.DataFrame()
    y_hat_test_df = pd.DataFrame()
    y_hat_train_df['ID'] = train_IDs
    y_hat_train_df['pred'] = y_hat_train
    y_hat_test_df['ID'] = test_IDs
    y_hat_test_df['pred'] = y_hat_test

    y_proba_train_df = pd.DataFrame()
    y_proba_test_df = pd.DataFrame()
    y_proba_train_df['ID'] = train_IDs
    y_proba_train_df['prob0'] = y_proba_train[:, 0]
    y_proba_train_df['prob1'] = y_proba_train[:, 1]
    y_proba_test_df['ID'] = test_IDs
    y_proba_test_df['prob0'] = y_proba_test[:, 0]
    y_proba_test_df['prob1'] = y_proba_test[:, 1]

    # Group by 'ID' and determine the average probability
    y_proba_train_mean = y_proba_train_df.groupby('ID')[['prob0', 'prob1']].mean().reset_index()
    y_proba_test_mean = y_proba_test_df.groupby('ID')[['prob0', 'prob1']].mean().reset_index()

    y_train_groupby_sorted = y_train_groupby.sort_values(by='ID')
    y_test_groupby_sorted = y_test_groupby.sort_values(by='ID')
    y_train_groupby_sorted.drop(columns=['ID'], inplace=True)
    y_test_groupby_sorted.drop(columns=['ID'], inplace=True)

    # Calculate AUC
    auc_roc_train = roc_auc_score(y_train_groupby_sorted.loc[:, target], y_proba_train_mean.loc[:, 'prob1'])
    auc_roc_test = roc_auc_score(y_test_groupby_sorted.loc[:, target], y_proba_test_mean.loc[:, 'prob1'])

    print(f'Split {i+1}/{num_tt_splits} AUC train: {auc_roc_train:.2f} AUC test: {auc_roc_test:.2f} Best Parameters:{best_params}')

    roc_auc_train.append(auc_roc_train)
    roc_auc_test.append(auc_roc_test)

    # Assign categories to the train and test predictions based on average probabilities
    y_hat_train_final_cat = y_proba_train_mean['prob1'].apply(lambda x: 1 if x > 0.5 else 0)
    y_hat_test_final_cat = y_proba_test_mean['prob1'].apply(lambda x: 1 if x > 0.5 else 0)

    if show_confusion_matrices:
        # Plot confusion matrices
        plot_confusion_matrix(y_train_groupby_sorted, y_hat_train_final_cat, 'Train')
        plot_confusion_matrix(y_test_groupby_sorted, y_hat_test_final_cat, 'Test')

    # Return logistic regression coefficients
    logreg_coef = best_model.named_steps['logreg'].coef_[0]  # Coefficients assigned by Logistic Regression

param_counter = Counter(best_parameters_list)

most_common_params, frequency = param_counter.most_common(1)[0]

most_common_params_dict = dict(most_common_params)

# Save parameters and scores to dataframe to find average auc_roc for most common parameters
data = {
    'best_parameters': best_parameters_list,
    'roc_auc_train': roc_auc_train,
    'roc_auc_test': roc_auc_test
}

param_score_df = pd.DataFrame(data)
param_score_df[['Cval', 'penaltyval', 'n_componentsval']] = param_score_df['best_parameters'].apply(pd.Series)
param_score_df.drop(columns=['best_parameters'], inplace=True)
for val in ['C', 'penalty', 'n_components']:
    param_score_df[val] = param_score_df[f'{val}val'].apply(lambda x: x[1])
param_score_df = param_score_df.drop(columns=['Cval', 'penaltyval', 'n_componentsval'])

avg_auc_train = param_score_df.loc[(param_score_df['C'] == most_common_params_dict['logreg__C'])
                               & (param_score_df['penalty'] == most_common_params_dict['logreg__penalty'])
                               & (param_score_df['n_components'] == most_common_params_dict['pca__n_components']),
                               'roc_auc_train'].mean()

avg_auc_test = param_score_df.loc[(param_score_df['C'] == most_common_params_dict['logreg__C'])
                               & (param_score_df['penalty'] == most_common_params_dict['logreg__penalty'])
                               & (param_score_df['n_components'] == most_common_params_dict['pca__n_components']),
                               'roc_auc_test'].mean()

print(f'Most common parameter combination: {most_common_params_dict} frequency {frequency}/{num_tt_splits}')
print(f'Avg AUC_ROC_train for most common param combination: {avg_auc_train}, Avg AUC_ROC_test: {avg_auc_test}')

# Final model with chosen hyperparameters
final_pipe_logreg_model = Pipeline([
    ('t', transformer),
    ('pca', PCA(n_components=most_common_params_dict['pca__n_components'])),
    ('logreg', LogisticRegression(penalty=most_common_params_dict['logreg__penalty'],
                                  C=most_common_params_dict['logreg__C'], solver='liblinear'))
])

# Fit the final model on the entire dataset
# Make sure data is transformed first
final_pipe_logreg_model.fit(ferret_cv[feature_columns], ferret_cv[target].values.ravel())

# Evaluate performance on TH animals
# Find all cases where animal is TH
ferret_th = ferret[ferret['Group'].isin([3])]
ferret_th.reset_index(inplace=True, drop=True)
y_th = ferret_th[target]
y_hat_th = final_pipe_logreg_model.predict(ferret_th[feature_columns])
y_proba_th = final_pipe_logreg_model.predict_proba(ferret_th[feature_columns])

th_IDs = ferret_th['ID']

# Assign ID numbers to predictions
y_hat_th_df = pd.DataFrame()
y_hat_th_df['ID'] = th_IDs
y_hat_th_df['pred'] = y_hat_th
y_hat_th_df.reset_index(inplace=True, drop=True)

y_proba_th_df = pd.DataFrame()
y_proba_th_df['ID'] = th_IDs
y_proba_th_df['prob0'] = y_proba_th[:, 0]
y_proba_th_df['prob1'] = y_proba_th[:, 1]

# Group by 'ID' and determine the average probability for th
y_proba_th_mean = y_proba_th_df.groupby('ID')[['prob0', 'prob1']].mean().reset_index()

# Make a dataframe of unique subjects with ID and Sex
th_groupby_df = ferret_th.groupby('ID').mean()
th_groupby_df = th_groupby_df.reset_index()
th_groupby_df = th_groupby_df[['ID', target]]
th_groupby_df.reset_index(inplace=True, drop=True)


# Caclulate AUC
auc_roc_th = roc_auc_score(th_groupby_df.loc[:, target], y_proba_th_mean.loc[:, 'prob1'])
print(f'AUC ROC for TH data is {auc_roc_th}')


mystop=1