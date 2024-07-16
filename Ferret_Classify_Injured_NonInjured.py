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

# Plot correlation matrix
ferret_to_corr = ferret_cv[feature_columns]
ferret_to_corr = ferret_to_corr.drop(columns=['Sex'])
corr_matrix = ferret_to_corr.corr()

# Retain upper triangular values of correlation matrix and
# make Lower triangular values Null
# make a mask for the upper triangle
mask = np.tril(np.ones(corr_matrix.shape, dtype=bool), k=-1)
# Apply mask to correlation matrix
lower_tri_corr = corr_matrix.where(mask)

# plt.figure(figsize=(16,16))
# sns.heatmap(lower_tri_corr, vmin=-1, vmax=1, xticklabels=lower_tri_corr.columns, yticklabels=lower_tri_corr.columns,
#             cmap='coolwarm')
# plt.tight_layout()
# plt.show(block=False)

# Find features that are not highly correlated with other features
s=lower_tri_corr.unstack().dropna()
s=s.reset_index()
s.rename(columns={0: 'rval'}, inplace=True)
s.sort_values(by='rval', ascending=False, inplace=True, ignore_index=True)
high_corr_features = s.loc[abs(s['rval']) > 0.7].copy()
high_corr_features.sort_values(by='level_1', inplace=True, ignore_index=True)
corr_features_to_remove = high_corr_features['level_1'].unique().tolist()
final_features = [f for f in feature_columns if f not in corr_features_to_remove]
final_columns = [col for col in ferret_cv if col not in corr_features_to_remove]

ferret_cv = ferret_cv[final_columns].copy()

# Plot new correlation matrix with highly correlated features removed
new_corr = ferret_cv[final_features]
new_corr = new_corr.drop(columns=['Sex'])
new_corr_matrix = new_corr.corr()
mask = np.tril(np.ones(new_corr_matrix.shape, dtype=bool), k=-1)
lower_tri_corr_nodup = new_corr_matrix.where(mask)

# plt.figure(figsize=(16,16))
# sns.heatmap(lower_tri_corr_nodup, vmin=-1, vmax=1, xticklabels=lower_tri_corr_nodup.columns,
#             yticklabels=lower_tri_corr_nodup.columns, cmap='coolwarm')
# plt.tight_layout()
# plt.show(block=False)

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

X_train_groupby, X_test_groupby, y_train_groupby, y_test_groupby = train_test_split(X, y, stratify=y[target], test_size=0.2, random_state=42)
# X_train_groupby, X_test_groupby, y_train_groupby, y_test_groupby = train_test_split(X, y, stratify=stratify_by, test_size=0.2, random_state=42)
X_train = ferret_cv.loc[ferret_cv['ID'].isin(X_train_groupby['ID']), final_features]
X_test = ferret_cv.loc[ferret_cv['ID'].isin(X_test_groupby['ID']), final_features]
y_train = ferret_cv.loc[ferret_cv['ID'].isin(y_train_groupby['ID']), target]
y_test = ferret_cv.loc[ferret_cv['ID'].isin(y_test_groupby['ID']), target]

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# create column transformer for standard scalar
# choose all columns except for the first column (sex)
columns_to_scale = final_features.copy()
columns_to_scale.remove('Sex')
transformer = ColumnTransformer(transformers=[('scale', StandardScaler(), columns_to_scale)],
                                remainder='passthrough')

# Create the pipeline for logistic regression (use for total gross score which was made binary)
pipe_logreg = Pipeline([
    ('t', transformer),
    ('pca', PCA(n_components=32)),
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

explained_variance = best_model.named_steps['pca'].explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)
n_components_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1

# predict values for the train and test data
y_hat_train = best_model.predict(X_train)
y_hat_test = best_model.predict(X_test)
y_proba_train = best_model.predict_proba(X_train)
y_proba_test = best_model.predict_proba(X_test)

auc_roc_train = roc_auc_score(y_train, y_proba_train[:, 1])
auc_roc_test = roc_auc_score(y_test, y_proba_test[:, 1])

# Calculate ROC curve
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_proba_train[:, 1])
roc_auc_train = auc(fpr_train, tpr_train)
fpr, tpr, thresholds = roc_curve(y_test, y_proba_test[:, 1])
roc_auc = auc(fpr, tpr)
# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve test set(area = %0.2f)' % roc_auc)
plt.plot(fpr_train, tpr_train, label='ROC curve train set (area=%.2f)' % roc_auc_train)
plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Injured Animal Identification')
plt.legend()
plt.show(block=False)

# Plot confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_train, y_hat_train)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix Train Data')
plt.show(block=False)

# Plot confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_hat_test)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix Test Data')
plt.show()

mystop=1