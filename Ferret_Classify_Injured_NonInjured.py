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

# Choose target variable
target = 'total gross score' #options: 'Pathology Score', 'total gross score', 'avg_5.30', 'Overall Sulci Sum', 'Overall Gyri Sum'
# Define project directory location
outputdir = '/home/toddr/neva/PycharmProjects/WoodAnimalData'

# Load ferret data with multiple run information
ferret_orig = pd.read_csv('data/Ferret CatWalk EpoTH IDs 60-74 Run Statistics with Brain Morphology.csv')

# Clean data
ferret = clean_data(ferret_orig)

# Plot distributions of features
# plot_feature_distributions(ferret, ferret.columns)

# Remove all cases where group is not control or vehicle
ferret_cv = ferret[ferret['Group'].isin([1,2])]
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

# Find and remove highly correlated features
final_features, final_columns = remove_highly_corr_features(ferret_cv, feature_columns, lower_tri_corr, rthreshold=0.7)
ferret_cv = ferret_cv[final_columns].copy()

# Calculate new lower triangle correlation matrix with highly correlated features removed
new_corr = ferret_cv[final_features]
new_corr = new_corr.drop(columns=['Sex'])
new_lower_tri_corr = calc_corr(new_corr, plotheatmap=0)

###################################################################################################
########################### SPLIT DATA INTO TRAIN TEST AND NORMALIZE ##############################
###################################################################################################

# Split the data into 80/20 train-test split stratified by the target column
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

# r = X['Sex'].corr(y[target])

X_train_groupby, X_test_groupby, y_train_groupby, y_test_groupby = train_test_split(X, y, stratify=y[target], test_size=0.2, random_state=1)
# X_train_groupby, X_test_groupby, y_train_groupby, y_test_groupby = train_test_split(X, y, stratify=stratify_by, test_size=0.2, random_state=42)

X_train = ferret_cv.loc[ferret_cv['ID'].isin(X_train_groupby['ID']), final_features]
X_test = ferret_cv.loc[ferret_cv['ID'].isin(X_test_groupby['ID']), final_features]
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

# create column transformer for standard scalar
# choose all columns except for the first column (sex)
columns_to_scale = final_features.copy()
columns_to_scale.remove('Sex')
transformer = ColumnTransformer(transformers=[('scale', StandardScaler(), columns_to_scale)],
                                remainder='passthrough')

###################################################################################################
########################### TRAIN AND FIT PCA AND LOGISTIC REGRESSION #############################
###################################################################################################

# Create the pipeline for logistic regression (use for total gross score which was made binary)
pipe_logreg = Pipeline([
    ('t', transformer),
    # ('pca', PCA(n_components=32)),
    ('logreg', LogisticRegression(solver='liblinear', random_state=42))
])

# Define parameter grid
param_grid = {
    'logreg__penalty': ['l1', 'l2'],
    'logreg__C': np.logspace(-3, 1, 5)
}

grid_search = GridSearchCV(pipe_logreg, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=4)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Parameters:", best_params)

# explained_variance = best_model.named_steps['pca'].explained_variance_ratio_
# cumulative_explained_variance = np.cumsum(explained_variance)
# n_components_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1

###################################################################################################
########################## MAKE CLASS PREDICTIONS AND EVALUATE PERFORMANCE ########################
###################################################################################################

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

auc_roc_train = roc_auc_score(y_train_groupby_sorted.loc[:, target], y_proba_train_mean.loc[:, 'prob1'])
auc_roc_test = roc_auc_score(y_test_groupby_sorted.loc[:, target], y_proba_test_mean.loc[:, 'prob1'])

# Assign categories to the train and test predictions based on average probabilities
y_hat_train_final_cat = y_proba_train_mean['prob1'].apply(lambda x: 1 if x > 0.5 else 0)
y_hat_test_final_cat = y_proba_test_mean['prob1'].apply(lambda x: 1 if x > 0.5 else 0)

# Compute AUC ROC and plot ROC curve for train and test set
calculate_roc_auc_and_plot(y_train_groupby_sorted.loc[:, target], y_test_groupby_sorted.loc[:, target],
                           y_proba_train_mean.loc[:, 'prob1'], y_proba_test_mean.loc[:, 'prob1'])

# Plot confusion matrices
plot_confusion_matrix(y_train_groupby_sorted, y_hat_train_final_cat, 'Train')
plot_confusion_matrix(y_test_groupby_sorted, y_hat_test_final_cat, 'Test')

# Extract PCA components and coefficients
pca_components = best_model.named_steps['pca'].components_  # Principal axes in feature space
logreg_coef = best_model.named_steps['logreg'].coef_[0]  # Coefficients assigned by Logistic Regression

# Find the largest logistic regression coefficients
abs_logreg_coef = np.abs(logreg_coef)

# Get indices of components sorted by absolute coefficient (descending order)
sorted_indices = np.argsort(abs_logreg_coef)[::-1]

# Print or plot the most important components
n_components_to_show = 5  # Adjust as needed
for i in range(n_components_to_show):
    component_idx = sorted_indices[i]
    print(f"PCA Component {component_idx + 1}: Coefficient {logreg_coef[component_idx]}")

original_feature_names = X_train.columns

# Get the indices of the top features contributing to each principal component
top_features_indices = np.argsort(np.abs(pca_components), axis=1)[:, ::-1]

# Print or store the top features for each principal component
n_top_features = 5  # Number of top features to display, adjust as needed
for i in range(5):
# for i in range(pca_components.shape[0]):  # Loop through each principal component
    print(f"Principal Component {i + 1}:")
    for j in range(n_top_features):  # Display the top features
        feature_idx = top_features_indices[i, j]
        print(f"- Feature: {original_feature_names[feature_idx]} (Loading: {pca_components[i, feature_idx]:.3f})")
    print()

mystop=1