import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def run_log_reg_on_ttsplit(stratified_split, X, y, target, n_splits, pipe_logreg):

    stratvar = y

    # Lists to collect performance and parameters for all splits
    auc_roc_train = np.zeros(n_splits)
    auc_roc_test = np.zeros(n_splits)

    splitnum = 0

    for train_index, test_index in stratified_split.split(X,  stratvar):
        X_train, X_test = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()
        y_train, y_test = y.iloc[train_index].copy(), y.iloc[test_index].copy()

        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        # Fit the model to the training data

        pipe_logreg.fit(X_train, y_train[target].ravel())
        # explained_variance = pipe_logreg.named_steps['pca'].explained_variance_ratio_
        # cumulative_explained_variance = np.cumsum(explained_variance)
        # n_components_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1

        # predict values for the train and test data
        y_hat_train = pipe_logreg.predict(X_train)
        y_hat_test = pipe_logreg.predict(X_test)
        auc_roc_train[splitnum] = roc_auc_score(y_train, y_hat_train)
        auc_roc_test[splitnum] = roc_auc_score(y_test, y_hat_test)
        # y_proba_train = pipe_logreg.predict_proba(X_train)
        # y_proba_test = pipe_logreg.predict_proba(X_test)
        # fpr, tpr, _ = roc_curve(y_train, y_proba_train[:,1])

        splitnum = splitnum + 1

    # Calculate average auc_roc across all folds
    avg_auc_roc_train = np.mean(auc_roc_train)
    avg_auc_roc_test = np.mean(auc_roc_test)

    return avg_auc_roc_train, avg_auc_roc_test

def run_log_reg_on_ttsplit_with_treat(X, y, target, pipe_logreg, X_treat, y_treat):

    # Fit the model to the control and vehicle data

    pipe_logreg.fit(X, y[target].ravel())

    #predict values for treatment data
    y_hat_treatment = pipe_logreg.predict(X_treat)

    #calculate ratio of two classes for treatment data
    percent_uninjured = np.count_nonzero(y_hat_treatment == 0)/len(y_hat_treatment)

    # Access the coefficients from the logistic regression step
    coefficients = pipe_logreg.named_steps['logreg'].coef_

    # Sum up all the logisitic regression coefficients
    coefficients_sum = coefficients.sum()

    return percent_uninjured, coefficients_sum