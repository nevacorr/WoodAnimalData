
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_curve, confusion_matrix

def clean_data(ferret_orig):
    ferret = ferret_orig.copy()

    # Brain regional size measurements
    columns_brain_volumes = ['total volume (cm^3)', 'cerebrum+brainstem (cm^3)', 'cerebellum (cm^3)', '% cerebellum',
                             'Summed White Matter GFAP (um)',
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

    # Find missing data count number of rows that contain string '-'
    count = (ferret == '-').any(axis=1).sum()
    # replace '-' with nan
    ferret.replace('-', np.nan, inplace=True)

    # ## Count and print total number of null values for each column with nans
    nan_cols = ferret.isna().sum()
    columns_with_nans = nan_cols[nan_cols > 0]

    # remove rows with nans
    ferret.dropna(inplace=True)
    ferret.reset_index(inplace=True, drop=True)

    # Recode gender as numeric categorical feature M=1, F=0
    ferret['Sex'] = ferret['Sex'].replace('F', 0)  # female = 0
    ferret['Sex'] = ferret['Sex'].replace('M', 1)  # male = 1

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

    # Check column datatypes
    dt = ferret.dtypes

    return ferret

def calc_corr(df, plotheatmap=0):
    corr_matrix = df.corr()

    # Retain upper triangular values of correlation matrix and
    # make Lower triangular values Null
    # make a mask for the upper triangle
    mask = np.tril(np.ones(corr_matrix.shape, dtype=bool), k=-1)
    # Apply mask to correlation matrix
    lower_tri_corr = corr_matrix.where(mask)

    if plotheatmap:
        plt.figure(figsize=(16, 16))
        sns.heatmap(lower_tri_corr, vmin=-1, vmax=1, xticklabels=lower_tri_corr.columns,
                    yticklabels=lower_tri_corr.columns,
                    cmap='coolwarm')

        # Adjust font size for the tick labels
        plt.tick_params(axis='x', labelsize=8)
        plt.tick_params(axis='y', labelsize=8)

        plt.tight_layout()
        plt.show(block=False)

    return lower_tri_corr

def remove_highly_corr_features(df, feature_columns, lower_tri_corr,  rthreshold):
# Find features that are not highly correlated with other features
    s=lower_tri_corr.unstack().dropna()
    s=s.reset_index()
    s.rename(columns={0: 'rval'}, inplace=True)
    s.sort_values(by='rval', ascending=False, inplace=True, ignore_index=True)
    high_corr_features = s.loc[abs(s['rval']) > rthreshold].copy()
    high_corr_features.sort_values(by='level_1', inplace=True, ignore_index=True)
    corr_features_to_remove = high_corr_features['level_1'].unique().tolist()
    final_features = [f for f in feature_columns if f not in corr_features_to_remove]
    final_columns = [col for col in df if col not in corr_features_to_remove]
    return final_features, final_columns

def calculate_roc_auc_and_plot(y_train, y_test, y_proba_train, y_proba_test, show_roc):

    # Calculate ROC curve
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_proba_train)
    roc_auc_train = auc(fpr_train, tpr_train)
    fpr_test, tpr_test, thresholds = roc_curve(y_test, y_proba_test)
    roc_auc_test = auc(fpr_test, tpr_test)
    if show_roc:
        # Plot the ROC curve
        plt.figure()
        plt.plot(fpr_test, tpr_test, label='ROC curve test set(area = %0.2f)' % roc_auc_test)
        plt.plot(fpr_train, tpr_train, label='ROC curve train set (area=%.2f)' % roc_auc_train)
        plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Injured Animal Identification')
        plt.legend()
        plt.show(block=False)
    return roc_auc_train, roc_auc_test

def plot_confusion_matrix(y, y_pred, titlestr):
    conf_matrix = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix {titlestr} Data')
    plt.show(block=False)

def return_most_important_features(coef, X_train):
    # Find the largest logistic regression coefficients
    abs_coef = np.abs(coef)

    # Get indices of feature sorted by absolute coefficient (descending order)
    sorted_indices = np.argsort(abs_coef)[::-1]

    # Count number of features with non zero coefficients
    non_zero_count = np.count_nonzero(abs_coef)

    important_features = []
    # Print the most important features
    for i in range(non_zero_count):
        feature_idx = sorted_indices[i]
        print(f"Feature importance #{i+1} Column #{feature_idx} {X_train.columns[feature_idx]}: Coefficient {coef[feature_idx]}")
        important_features.append(X_train.columns[feature_idx])

    return important_features