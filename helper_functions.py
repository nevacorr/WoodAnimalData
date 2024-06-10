#helper functions#

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LassoCV

def get_dataframe_info(df):
    """
    input
       df -> DataFrame
    output
       df_null_counts -> DataFrame Info (sorted)
    """

    df_types = pd.DataFrame(df.dtypes)
    df_nulls = df.count()

    df_null_count = pd.concat([df_types, df_nulls], axis=1)
    df_null_count = df_null_count.reset_index()

    # Reassign column names
    col_names = ["features", "types", "non_null_counts"]
    df_null_count.columns = col_names

    # Add this to sort
    df_null_count = df_null_count.sort_values(by=["non_null_counts"], ascending=False)

    return df_null_count

def remove_columns_with_substring(dataframe, substring):
    columns_to_drop = [col for col in dataframe.columns if substring in col]
    dataframe = dataframe.drop(columns=columns_to_drop)
    return dataframe

def plot_feature_distributions(data_df, column_list):
    df = data_df[column_list].copy()

    #Plot distributions
    n_rows=5
    n_cols=6

    sns.set(font_scale=0.5)
    num_df_cols=df.shape[1]
    m=math.ceil(num_df_cols/30)*30

    for row in range(0,m,30):
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
        fig.subplots_adjust(hspace=0.3, wspace=0.5)
        fig.set_size_inches(12, 15)
        for i, column in enumerate(column_list[row:row+30]):
            sns.histplot(df[column],ax=axes[i//n_cols,i%n_cols])
        plt.show(block=False)
        mystop=1

def calculate_and_replace_outliers(data_df, column_list):
    df=data_df[column_list].copy()
    outfile=open('rat_data_features_with_outliers_removed.txt', 'w')
    outfile.write('Behavior, numlow_outliers, numhigh_outliers\n')
    for col in column_list:
        # Calculate the upper and lower limits
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Create arrays of Boolean values indicating the outlier rows
        upper_array = np.where(df[col] >= upper)[0]
        lower_array = np.where(df[col] <= lower)[0]

        #replace outliers with median value
        if len(upper_array>0) | len(lower_array)>0:
            med=df[col].quantile(0.50)
            df[col] = np.where(df[col] >= upper, med, df[col])
            df[col] = np.where(df[col] <= lower, med, df[col])
            str_to_write=col + ',' + str(len(lower_array)) + ',' + str(len(upper_array)) + '\n'
            outfile.write(str_to_write)
    outfile.close()
    stop=1

def replace_nan_with_median(df, column_name):
        median_value = df[column_name].median()
        df[column_name].fillna(median_value, inplace=True)
        return df

def plot_boxplots(data_df, column_list):
    df = data_df[column_list].copy()
    sns.set(font_scale=0.5)
    num_df_cols = len(column_list)
    num_subplots_per_figure = 30

    num_figures = math.ceil(num_df_cols / num_subplots_per_figure)

    for fig_num in range(num_figures):
        fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(16, 20))
        start_index = fig_num * num_subplots_per_figure
        end_index = start_index + num_subplots_per_figure
        subplot_columns = min(num_df_cols - start_index, num_subplots_per_figure)

        for i, col in enumerate(column_list[start_index:end_index]):
            ax = axes[i // 6, i % 6]  # Get the appropriate axis for the subplot
            df.boxplot(column=col, ax=ax)
  #          ax.set_title(col)  # Set the title of the boxplot
            ax.set_xlabel('')  # Remove the xlabel to avoid overlap
            ax.set_ylabel('')  # Remove the ylabel to avoid overlap

        plt.tight_layout()  # Adjust spacing between subplots
        plt.show()

def plot_expvar_vs_numcomponents_return_best(explained_variance_ratios):
    df = pd.DataFrame()
    for i in range(len(explained_variance_ratios)):
        df.loc[i, 'num_pcs'] = i
        df.loc[i, 'cumulative_sum'] = sum(explained_variance_ratios[:i])
    optimal_ncomponents = np.argmax(df['cumulative_sum'].to_numpy() >= 0.9500000)
    df.plot(x="num_pcs", y=["cumulative_sum"], marker='o',figsize=(15, 8))
    plt.axhline(y=0.95, color='red')
    plt.grid(True)
    plt.xticks(range(1, explained_variance_ratios.shape[0]))
    plt.title(' Explained variances vs. Num Components ')
    plt.show(block=False)
    return optimal_ncomponents

def find_non_numeric_columns(df):
    non_numeric_columns = []
    for column in df.columns:
        try:
            pd.to_numeric(df[column])
        except ValueError:
            non_numeric_columns.append(column)
    return non_numeric_columns

def find_non_numeric_rows(df, columns):
    non_numeric_rows = {}
    for col in columns:
        non_numeric_rows[col] = df.loc[pd.to_numeric(df[col], errors='coerce').isna()]
    return non_numeric_rows

def plot_correlation_between_features_and_pcfeatures(X_train, X_train_pca_features):
    X_train.reset_index(inplace=True, drop=True)
    features_pca = ['pc_' + str(i+1) for i in range(X_train_pca_features.shape[1])]
    df_pca_features = pd.DataFrame(data=X_train_pca_features, columns = features_pca)
    newdf = pd.concat([X_train, df_pca_features], axis=1, ignore_index=True)
    newdf.columns = X_train.columns.tolist() + features_pca
    corr_matrix=newdf.corr()
    X_train_feature_names = X_train.columns.tolist()
    corr_matrix.drop(columns=X_train_feature_names, inplace=True)
    corr_matrix.drop(index=features_pca, inplace=True)
    fig, ax = plt.subplots(figsize=(10,12))
    sns.heatmap(corr_matrix,  cmap='coolwarm')
    plt.title('Correlations between Original Features and Principal Components')
    plt.subplots_adjust(left=0.4)
    plt.show()

def scatter_plot_lda_transformed_2d(X, y, target_names, datastr):
    plt.figure(figsize=(10,6))
    plt.subplots_adjust(right=0.8)
    colors = ['red', 'green', 'blue']
    lw=2
    for color, i, target_name in zip(colors, [-1, 0, 1], target_names):
        plt.scatter(X[y==i, 0], X[y==i, 1], alpha=0.8, color=color, label=target_name)
    plt.legend(loc='center left', scatterpoints=1, shadow=False, bbox_to_anchor=(1,0.5))
    plt.title(f'Scatterplot of LDA results for Raw Ferret {datastr} Data')
    plt.xlabel('Discriminant Component 1')
    plt.ylabel('Discriminant Component 2')
    plt.tight_layout
    plt.show(block=False)

def plot_feature_importance(feature_coefficients, feature_columns, target_names):
    # Specify the number of top features to show
    top_features = 10

    # Iterate over each class
    for class_index, class_coefficients in enumerate(feature_coefficients):
        # Get the top feature indices for the current class
        top_indices = np.argsort(np.abs(class_coefficients))[::-1][:top_features]

        # Get the corresponding coefficients and feature names
        top_coefficients = class_coefficients[top_indices][::-1] #Reverse the order of coefficients
        top_feature_names = np.array(feature_columns)[top_indices][::-1]# Reverse the order of feature names

        # Create a bar plot for the current class
        plt.figure()
        plt.barh(range(top_features), top_coefficients, align='center')
        plt.yticks(range(top_features), top_feature_names)
        plt.xlabel('Coefficient')
        plt.ylabel('Features')
        plt.title(f'Top Features for Class {target_names[class_index]}')
        plt.tight_layout()
        plt.show(block=False)

def plot_lda_coeff(feature_coefficients, feature_columns):
    plt.figure()
    coefficients = feature_coefficients
    feature_names = feature_columns
    plt.bar(feature_names, coefficients[0])
    plt.bar(feature_names, coefficients[1])
    plt.bar(feature_names, coefficients[2])
    plt.xlabel('Features')
    plt.ylabel('Coefficient Values')
    plt.title(f'LDA Coefficients')
    plt.xticks(rotation=45)
    plt.show(block=False)

def get_accuracy_by_class(y_true, y_pred, classes):
    # Get the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # We will store the results in a dictionary for easy access later
    per_class_accuracies = {}

    # Calculate the accuracy for each one of our classes
    for idx, cls in enumerate(classes):
        # True negatives are all the samples that are not our current  class (not the current row)
        # and were not predicted as the current class (not the current column)
        true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))

        # True positives are all the samples of our current  class that were predicted as such
        true_positives = cm[idx, idx]

        # The accuracy for the current class is the ratio between correct predictions to all predictions
        per_class_accuracies[cls] = (true_positives + true_negatives) / np.sum(cm)
    return per_class_accuracies

def plot_lasso_coeff_by_alpha(X, y):
    alphas=np.logspace(-120, 12, 200)
    lasso = LassoCV()
    coefs = []

    for a in alphas:
        lasso.set_params(alphas=alphas)
        lasso.fit(X, y)
        coefs.append(lasso.coef_)

    plt.figure(figsize=(40, 20))
    plt.subplot(121)
    ax=plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.axis('tight')
    plt.show()

def plot_proba_histograms(probs, label_list, color_list, modelstr, testtrainstr):
    for i, prob_list in enumerate(probs):
        plt.hist(prob_list, alpha=0.3, label=label_list[i], color=color_list[i], density=False)
    plt.legend(loc='upper right')
    plt.title(f'Probability of being classified as vehicle by {modelstr} model ' + testtrainstr)
    plt.xlabel('probability of being classified as vehicle ')
    plt.ylabel('number of cases')
    plt.show()

def plot_y_v_yhat(y, y_hat, data_type, model_type, measure, r2):
    plt.figure()
    ax = plt.axes()
    #plot data
    sc = ax.scatter(y, y_hat)
    # plots line y = x
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
    ax.set_xlabel(f'{measure}')
    ax.set_ylabel(f'predicted {measure}]')
    ax.set_title(f'{model_type} predicted vs actual {measure}\n for {data_type} data coefficient of determination = {r2:.2f}')
    plt.show(block=False)

def write_list_to_file(mylist, filename):
    file=open(filename, 'w')
    for item in mylist:
        file.write(item+"\n")
    file.close()
