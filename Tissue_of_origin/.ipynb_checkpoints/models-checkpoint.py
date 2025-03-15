from plotting_utils import plot_classification_results, plot_roc_curve, plot_confusion_matrix, plot_separate_normalized_confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, make_scorer, roc_auc_score, accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
import joblib

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score


import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from sklearn.metrics import  precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef
import matplotlib.pyplot as plt
import os
import csv

def train_model_nested_cv(dataset, target_name='target', model=None, param_grid=None, search_method='grid', scoring=None, save_folder=None, outer_cv_folds=5, inner_cv_folds=3):
    """
    Trains a model using Nested Cross-Validation, plots the ROC curve, and confusion matrix.

    Parameters:
        dataset (DataFrame): Combined dataset with features and target column.
        target_name (str): The name of the target column.
        model (object): Machine learning model to train. Defaults to RandomForestClassifier.
        param_grid (dict): Hyperparameter grid for tuning. If None, no tuning is performed.
        search_method (str): Method for hyperparameter search ('grid' or 'random').
        scoring (str): Metric for model evaluation (e.g., 'roc_auc_macro', 'accuracy').
        save_folder (str): Folder to save ROC data files. If None, files are not saved.
        outer_cv_folds (int): Number of folds for the outer cross-validation.
        inner_cv_folds (int): Number of folds for the inner cross-validation.

    Returns:
        model (object): The trained model.
        nested_cv_accuracy (float): The average accuracy from Nested Cross-Validation.
    """

    # If no model is provided, use RandomForestClassifier as the default
    if model is None:
        model = RandomForestClassifier(random_state=0, class_weight='balanced')

    # Splitting dataset into features and target
    X = dataset.drop(columns=[target_name])
    y = dataset[target_name]

    # Extract the class labels from y
    classes = np.unique(y)  # Automatically deduce class names from y

    # Configure the scoring metric based on user input
    if scoring == 'roc_auc_macro':
        scoring_metric = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='macro')
    elif scoring == 'roc_auc_micro':
        scoring_metric = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='micro')
    elif scoring == 'balanced_accuracy':
        scoring_metric = 'balanced_accuracy'
    else:
        # Default to accuracy if no or unrecognized scoring is provided
        scoring_metric = scoring or 'accuracy'

    # Set up outer cross-validation
    outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=0)

    nested_accuracies = []
    predictions = []
    probabilities = []
    true_labels = []

    # Outer loop for performance evaluation
    for train_index, test_index in outer_cv.split(X, y):
        # Split data into training and testing folds for outer CV
        X_train_outer, X_test_outer = X.iloc[train_index], X.iloc[test_index]
        y_train_outer, y_test_outer = y.iloc[train_index], y.iloc[test_index]

        # If hyperparameter tuning is requested
        if param_grid is not None:
            # Set up inner cross-validation for hyperparameter tuning
            inner_cv = StratifiedKFold(n_splits=inner_cv_folds, shuffle=True, random_state=0)

            if search_method == 'grid':
                search = GridSearchCV(model, param_grid, cv=inner_cv, scoring=scoring_metric, n_jobs=-1)
            elif search_method == 'random':
                search = RandomizedSearchCV(model, param_grid, cv=inner_cv, scoring=scoring_metric, n_jobs=-1, random_state=0)

            # Perform the search on the training folds
            search.fit(X_train_outer, y_train_outer)
            model = search.best_estimator_
            print(f"Best hyperparameters found for this fold: {search.best_params_}")

        # Train the model on the training data of the outer fold
        model.fit(X_train_outer, y_train_outer)

        # Predict on the test fold
        y_pred = model.predict(X_test_outer)
        y_proba = model.predict_proba(X_test_outer)

        predictions.extend(y_pred)
        probabilities.extend(y_proba)
        true_labels.extend(y_test_outer)

        # Evaluate accuracy for this fold
        fold_accuracy = accuracy_score(y_test_outer, y_pred)
        nested_accuracies.append(fold_accuracy)

    # Convert lists to numpy arrays
    probabilities = np.array(probabilities)
    true_labels = np.array(true_labels)

    # Calculate average nested CV accuracy
    nested_cv_accuracy = np.mean(nested_accuracies)
    print(f"Nested CV Accuracy: {nested_cv_accuracy:.4f}")

    # Train final model on the entire dataset
    model.fit(X, y)

    # Plot ROC Curve using existing function
    plot_roc_curve(pd.Series(true_labels), probabilities, y, target_name, classes, save_folder)

    # Plot Confusion Matrix using existing function
    plot_confusion_matrix(true_labels, predictions, target_name, classes)

    return model, nested_cv_accuracy




def combine_train_test(train_df, test_df):
    """
    Combines train and test datasets into a single dataset for unified processing.

    Parameters:
        train_df (DataFrame): Training data with features and target column.
        test_df (DataFrame): Test data with features and target column.

    Returns:
        combined_df (DataFrame): The combined dataset with train and test data.
    """
    combined_df = pd.concat([train_df, test_df], verify_integrity=True)
    return combined_df

# def train_model_loocv(dataset, target_name='target', model=None, param_grid=None, search_method='grid', scoring=None, save_folder=None,save_figures_path=None,output_file=None):
#     """
#     Trains a model using Leave-One-Out Cross-Validation (LOOCV) on a single dataset, plots the ROC curve, and confusion matrix.

#     Parameters:
#         dataset (DataFrame): Combined dataset with features and target column.
#         target_name (str): The name of the target column.
#         model (object): Machine learning model to train. Defaults to RandomForestClassifier.
#         param_grid (dict): Hyperparameter grid for tuning. If None, no tuning is performed.
#         search_method (str): Method for hyperparameter search ('grid' or 'random').
#         scoring (str): Metric for model evaluation (e.g., 'roc_auc_macro', 'accuracy').
#         save_folder (str): Folder to save ROC data files. If None, files are not saved.

#     Returns:
#         model (object): The trained model.
#         loocv_accuracy (float): The accuracy from LOOCV.
#     """

#     # If no model is provided, use RandomForestClassifier as the default
#     if model is None:
#         model = RandomForestClassifier(random_state=0, class_weight='balanced')

#     # Splitting dataset into features and target
#     X = dataset.drop(columns=[target_name])
#     y = dataset[target_name]

#     # Extract the class labels from y
#     classes = np.unique(y)  # Automatically deduce class names from y

#     # Configure the scoring metric based on user input
#     if scoring == 'roc_auc_macro':
#         scoring_metric = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='macro')
#     elif scoring == 'roc_auc_micro':
#         scoring_metric = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='micro')
#     elif scoring == 'balanced_accuracy':
#         scoring_metric = 'balanced_accuracy'
#     else:
#         # Default to accuracy if no or unrecognized scoring is provided
#         scoring_metric = scoring or 'accuracy'

#     # Set up Leave-One-Out Cross-Validation
#     loo = LeaveOneOut()

#     # If hyperparameter tuning is requested
#     if param_grid is not None:
#         if search_method == 'grid':
#             search = GridSearchCV(model, param_grid, cv=loo, scoring=scoring_metric, n_jobs=-1)
#         elif search_method == 'random':
#             search = RandomizedSearchCV(model, param_grid, cv=loo, scoring=scoring_metric, n_jobs=-1, random_state=0)

#         # Perform the search and find the best model
#         search.fit(X, y)
#         model = search.best_estimator_
#         print(f"Best hyperparameters found: {search.best_params_}")
    
#     # Perform LOOCV
#     predictions = []
#     probabilities = []
#     true_labels = []

#     for train_index, test_index in loo.split(X):
#         # Split data into training and testing folds for LOOCV
#         X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
#         y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

#         # Train the model on the training fold
#         model.fit(X_train_fold, y_train_fold)

#         # Predict on the test fold
#         y_pred = model.predict(X_test_fold)
#         y_proba = model.predict_proba(X_test_fold)

#         predictions.append(y_pred[0])  # Append the predicted label
#         probabilities.append(y_proba[0])  # Append the predicted probabilities
#         true_labels.append(y_test_fold.iloc[0])  # Append the true label

#     # Convert lists to numpy arrays
#     probabilities = np.array(probabilities)
#     true_labels = np.array(true_labels)

#     # Calculate LOOCV accuracy
#     loocv_accuracy = accuracy_score(true_labels, predictions)
#     print(f"LOOCV Accuracy: {loocv_accuracy:.4f}")


#     if save_figures_path:
#         # Temporarily suppress plt.show()
#         original_show = plt.show
#         plt.show = lambda: None
#          # Plot ROC Curve using existing function
#         plot_roc_curve(pd.Series(true_labels), probabilities, y, target_name, classes, save_folder)
#         plt.savefig(save_figures_path+"/cancer_roc.png")  # Save the loss vs epochs plot
#         plt.clf()  # Clear the plot

#         # Plot Confusion Matrix using existing function
#         plot_confusion_matrix(true_labels, predictions, target_name, classes)

#         plt.savefig(save_figures_path+"/cancer_confusion_matrix.png")  # Save the confusion matrix plot
#         plt.clf()  # Clear the plot


#         plot_separate_normalized_confusion_matrix(true_labels, predictions, target_name, classes)

#         plt.savefig(save_figures_path+"/cancer_confusion_matrix_Normalized.png")  # Save the confusion matrix plot
#         plt.clf()  # Clear the plot

#         # Restore plt.show
#         plt.show = original_show


#         if output_file:
#             # Determine whether the problem is binary or multi-class
#             if len(np.unique(true_labels)) == 2:  # Binary classification
#                 average_type = 'binary'
#                 pos_label = 'Bladder'  # Set 'Bladder' as the positive label
#                 probabilities_positive_class = probabilities[:, 1]  # Extract probabilities for the positive class
#                 auc = roc_auc_score(true_labels, probabilities_positive_class)
#             else:  # Multi-class classification
#                 average_type = 'macro'  # Adjust as needed
#                 pos_label = None
#                 auc = roc_auc_score(true_labels, probabilities, multi_class='ovr')

#             # Calculate precision, recall, and F1 score dynamically
#             precision = precision_score(true_labels, predictions, average=average_type, pos_label=pos_label)
#             recall = recall_score(true_labels, predictions, average=average_type, pos_label=pos_label)
#             f1 = f1_score(true_labels, predictions, average=average_type, pos_label=pos_label)

#             # Calculate additional metrics
#             balanced_acc = balanced_accuracy_score(true_labels, predictions)
#             mcc = matthews_corrcoef(true_labels, predictions)

#             # Save metrics to file
#             save_metrics_to_file(os.path.basename(save_figures_path), precision, recall, f1, auc, balanced_acc, mcc, output_file)




#     else:
#         # Plot ROC Curve using existing function
#         plot_roc_curve(pd.Series(true_labels), probabilities, y, target_name, classes, save_folder)

#         # Plot Confusion Matrix using existing function
#         plot_confusion_matrix(true_labels, predictions, target_name, classes)

    


       

#     # Train final model on the entire dataset
#     model.fit(X, y)
   

#     return model, loocv_accuracy


# def train_model_loocv(dataset, target_name='target', model=None, param_grid=None, search_method='grid', scoring=None, save_folder=None, save_figures_path=None, output_file=None):
#     """
#     Trains a model using Leave-One-Out Cross-Validation (LOOCV) on a single dataset, 
#     plots the ROC curve, and confusion matrix.

#     Returns:
#         model (object): The trained model.
#         loocv_accuracy (float): The accuracy from LOOCV.
#         oof_df (DataFrame): A DataFrame containing true labels, predicted labels, and class probabilities.
#     """

#     average_auc = -1
#     # If no model is provided, use RandomForestClassifier as the default
#     if model is None:
#         model = RandomForestClassifier(random_state=0, class_weight='balanced')

#     # Splitting dataset into features and target
#     X = dataset.drop(columns=[target_name])
#     y = dataset[target_name]
#     classes = np.unique(y)  # Automatically deduce class names from y

#     # Configure the scoring metric based on user input
#     if scoring == 'roc_auc_macro':
#         scoring_metric = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='macro')
#     elif scoring == 'roc_auc_micro':
#         scoring_metric = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='micro')
#     elif scoring == 'balanced_accuracy':
#         scoring_metric = 'balanced_accuracy'
#     else:
#         scoring_metric = scoring or 'accuracy'

#     # Set up Leave-One-Out Cross-Validation
#     loo = LeaveOneOut()

#     # If hyperparameter tuning is requested
#     if param_grid is not None:
#         if search_method == 'grid':
#             search = GridSearchCV(model, param_grid, cv=loo, scoring=scoring_metric, n_jobs=-1)
#         elif search_method == 'random':
#             search = RandomizedSearchCV(model, param_grid, cv=loo, scoring=scoring_metric, n_jobs=-1, random_state=0)

#         search.fit(X, y)
#         model = search.best_estimator_
#         print(f"Best hyperparameters found: {search.best_params_}")

#     # OOF storage with index preservation
#     oof_data = []  
#     indices = []  # Store original indices
    
#     for train_index, test_index in loo.split(X):
#         X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
#         y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    
#         model.fit(X_train_fold, y_train_fold)
    
#         y_pred = model.predict(X_test_fold)
#         y_proba = model.predict_proba(X_test_fold)
    
#         # Store the OOF results
#         row = {'True Label': y_test_fold.iloc[0], 'Predicted Label': y_pred[0]}
#         for class_idx, class_label in enumerate(classes):
#             row[f'Probability_{class_label}'] = y_proba[0, class_idx]
        
#         oof_data.append(row)
#         indices.append(X_test_fold.index[0])  # Store original index
    
#     # Convert to DataFrame and set indices
#     oof_df = pd.DataFrame(oof_data, index=indices)
    
#     # Ensure index is restored correctly
#     oof_df.index.name = 'Sample Index'

#     # Calculate LOOCV accuracy
#     loocv_accuracy = accuracy_score(oof_df['True Label'], oof_df['Predicted Label'])
#     print(f"LOOCV Accuracy: {loocv_accuracy:.4f}")

#     if save_figures_path:
#         original_show = plt.show
#         plt.show = lambda: None

#         # Plot ROC Curve using existing function
#         average_auc = plot_roc_curve(oof_df['True Label'], oof_df.iloc[:, 2:].values, y, target_name, classes, save_folder)
#         plt.savefig(save_figures_path+"/"+os.path.basename(save_figures_path)+"_cancer_roc.png")
#         plt.clf()

#         # Plot Confusion Matrix using existing function
#         plot_confusion_matrix(oof_df['True Label'], oof_df['Predicted Label'], target_name, classes)
#         plt.savefig(save_figures_path+"/"+os.path.basename(save_figures_path)+"_cancer_confusion_matrix.png")
#         plt.clf()

#         plot_separate_normalized_confusion_matrix(oof_df['True Label'], oof_df['Predicted Label'], target_name, classes)
#         plt.savefig(save_figures_path+"/"+os.path.basename(save_figures_path)+"_cancer_confusion_matrix_Normalized.png")
#         plt.clf()

#         plt.show = original_show

#         if output_file:
#             if len(np.unique(oof_df['True Label'])) == 2:  # Binary classification
#                 average_type = 'binary'
#                 pos_label = 'Bladder'
#                 probabilities_positive_class = oof_df[f'Probability_Bladder']
#                 auc = roc_auc_score(oof_df['True Label'], probabilities_positive_class)
#             else:  # Multi-class classification
#                 average_type = 'macro'
#                 pos_label = None
#                 auc = roc_auc_score(oof_df['True Label'], oof_df.iloc[:, 2:].values, multi_class='ovr')

#             precision = precision_score(oof_df['True Label'], oof_df['Predicted Label'], average=average_type, pos_label=pos_label)
#             recall = recall_score(oof_df['True Label'], oof_df['Predicted Label'], average=average_type, pos_label=pos_label)
#             f1 = f1_score(oof_df['True Label'], oof_df['Predicted Label'], average=average_type, pos_label=pos_label)
#             balanced_acc = balanced_accuracy_score(oof_df['True Label'], oof_df['Predicted Label'])
#             mcc = matthews_corrcoef(oof_df['True Label'], oof_df['Predicted Label'])

#             save_metrics_to_file(os.path.basename(save_figures_path), precision, recall, f1, auc, balanced_acc, mcc, output_file)

#     else:
#         plot_roc_curve(oof_df['True Label'], oof_df.iloc[:, 2:].values, y, target_name, classes, save_folder)
#         plot_confusion_matrix(oof_df['True Label'], oof_df['Predicted Label'], target_name, classes)

#     # Train final model on the entire dataset
#     model.fit(X, y)

#     return model, average_auc, oof_df


def train_model_loocv(dataset, target_name='target', model=None, param_grid=None, search_method='grid', scoring=None, save_folder=None, save_figures_path=None, output_file=None):
    """
    Trains a model using Leave-One-Out Cross-Validation (LOOCV) on a single dataset, 
    plots the ROC curve, and confusion matrix.

    Returns:
        model (object): The trained model.
        loocv_accuracy (float): The accuracy from LOOCV.
        oof_df (DataFrame): A DataFrame containing true labels, predicted labels, and class probabilities.
    """

    average_auc = -1
    # If no model is provided, use RandomForestClassifier as the default
    if model is None:
        model = RandomForestClassifier(random_state=0, class_weight='balanced')

    # Store the original indices before transforming
    original_indices = dataset.index
    
    # Apply StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(dataset.drop(columns=[target_name]))
    y = dataset[target_name]

    classes = np.unique(y)  # Automatically deduce class names from y

    # Configure the scoring metric based on user input
    if scoring == 'roc_auc_macro':
        scoring_metric = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='macro')
    elif scoring == 'roc_auc_micro':
        scoring_metric = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='micro')
    elif scoring == 'balanced_accuracy':
        scoring_metric = 'balanced_accuracy'
    else:
        scoring_metric = scoring or 'accuracy'

    # Set up Leave-One-Out Cross-Validation
    loo = LeaveOneOut()

    # If hyperparameter tuning is requested
    if param_grid is not None:
        if search_method == 'grid':
            search = GridSearchCV(model, param_grid, cv=loo, scoring=scoring_metric, n_jobs=-1)
        elif search_method == 'random':
            search = RandomizedSearchCV(model, param_grid, cv=loo, scoring=scoring_metric, n_jobs=-1, random_state=0)

        search.fit(X, y)
        model = search.best_estimator_
        print(f"Best hyperparameters found: {search.best_params_}")

    # OOF storage with index preservation
    oof_data = []  
    indices = []  # Store original indices
    
    for train_index, test_index in loo.split(X):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    
        model.fit(X_train_fold, y_train_fold)
    
        y_pred = model.predict(X_test_fold)
        y_proba = model.predict_proba(X_test_fold)
    
        # Store the OOF results
        row = {'True Label': y_test_fold.iloc[0], 'Predicted Label': y_pred[0]}
        for class_idx, class_label in enumerate(classes):
            row[f'Probability_{class_label}'] = y_proba[0, class_idx]
        
        oof_data.append(row)
        indices.append(original_indices[test_index][0])  # Retrieve the original index correctly
    
    # Convert to DataFrame and set indices
    oof_df = pd.DataFrame(oof_data, index=indices)
    
    # Ensure index is restored correctly
    oof_df.index.name = 'Sample Index'

    # Calculate LOOCV accuracy
    loocv_accuracy = accuracy_score(oof_df['True Label'], oof_df['Predicted Label'])
    print(f"LOOCV Accuracy: {loocv_accuracy:.4f}")

    if save_figures_path:
        original_show = plt.show
        plt.show = lambda: None

        # Plot ROC Curve using existing function
        average_auc = plot_roc_curve(oof_df['True Label'], oof_df.iloc[:, 2:].values, y, target_name, classes, save_folder)
        plt.savefig(save_figures_path+"/"+os.path.basename(save_figures_path)+"_cancer_roc.png")
        plt.clf()

        # Plot Confusion Matrix using existing function
        plot_confusion_matrix(oof_df['True Label'], oof_df['Predicted Label'], target_name, classes)
        plt.savefig(save_figures_path+"/"+os.path.basename(save_figures_path)+"_cancer_confusion_matrix.png")
        plt.clf()

        plot_separate_normalized_confusion_matrix(oof_df['True Label'], oof_df['Predicted Label'], target_name, classes)
        plt.savefig(save_figures_path+"/"+os.path.basename(save_figures_path)+"_cancer_confusion_matrix_Normalized.png")
        plt.clf()

        plt.show = original_show

        if output_file:
            if len(np.unique(oof_df['True Label'])) == 2:  # Binary classification
                average_type = 'binary'
                pos_label = 'Bladder'
                probabilities_positive_class = oof_df[f'Probability_Bladder']
                auc = roc_auc_score(oof_df['True Label'], probabilities_positive_class)
            else:  # Multi-class classification
                average_type = 'macro'
                pos_label = None
                auc = roc_auc_score(oof_df['True Label'], oof_df.iloc[:, 2:].values, multi_class='ovr')

            precision = precision_score(oof_df['True Label'], oof_df['Predicted Label'], average=average_type, pos_label=pos_label)
            recall = recall_score(oof_df['True Label'], oof_df['Predicted Label'], average=average_type, pos_label=pos_label)
            f1 = f1_score(oof_df['True Label'], oof_df['Predicted Label'], average=average_type, pos_label=pos_label)
            balanced_acc = balanced_accuracy_score(oof_df['True Label'], oof_df['Predicted Label'])
            mcc = matthews_corrcoef(oof_df['True Label'], oof_df['Predicted Label'])

            save_metrics_to_file(os.path.basename(save_figures_path), precision, recall, f1, auc, balanced_acc, mcc, output_file)

    else:
        plot_roc_curve(oof_df['True Label'], oof_df.iloc[:, 2:].values, y, target_name, classes, save_folder)
        plot_confusion_matrix(oof_df['True Label'], oof_df['Predicted Label'], target_name, classes)

    # Train final model on the entire dataset
    model.fit(X, y)

    model.scaler = scaler

    return model, average_auc, oof_df

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

def train_model_loocv_xgb(dataset, target_name='target', model=None, param_grid=None, search_method='grid', scoring=None, save_folder=None, save_figures_path=None, output_file=None):
    """
    Trains a model using Leave-One-Out Cross-Validation (LOOCV) on a single dataset, 
    plots the ROC curve, and confusion matrix.

    Returns:
        model (object): The trained model.
        loocv_accuracy (float): The accuracy from LOOCV.
        oof_df (DataFrame): A DataFrame containing true labels, predicted labels, and class probabilities.
    """

    # If no model is provided, use RandomForestClassifier as the default
    if model is None:
        model = RandomForestClassifier(random_state=0, class_weight='balanced')

    # Splitting dataset into features and target
    X = dataset.drop(columns=[target_name])
    y = dataset[target_name]

    # Encode target labels if they are categorical
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # Convert to numerical labels
    class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))  # Store mapping
    class_names = label_encoder.classes_  # Store class names for plotting

    # Configure the scoring metric
    if scoring == 'roc_auc_macro':
        scoring_metric = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='macro')
    elif scoring == 'roc_auc_micro':
        scoring_metric = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='micro')
    elif scoring == 'balanced_accuracy':
        scoring_metric = 'balanced_accuracy'
    else:
        scoring_metric = scoring or 'accuracy'

    # Set up Leave-One-Out Cross-Validation
    loo = LeaveOneOut()

    # If hyperparameter tuning is requested
    if param_grid is not None:
        if search_method == 'grid':
            search = GridSearchCV(model, param_grid, cv=loo, scoring=scoring_metric, n_jobs=-1)
        elif search_method == 'random':
            search = RandomizedSearchCV(model, param_grid, cv=loo, scoring=scoring_metric, n_jobs=-1, random_state=0)

        search.fit(X, y_encoded)  # Train on encoded labels
        model = search.best_estimator_
        print(f"Best hyperparameters found: {search.best_params_}")

    # OOF storage with index preservation
    oof_data = []  
    indices = []  # Store original indices
    
    for train_index, test_index in loo.split(X):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y_encoded[train_index], y_encoded[test_index]
    
        model.fit(X_train_fold, y_train_fold)
    
        y_pred = model.predict(X_test_fold)
        y_proba = model.predict_proba(X_test_fold)
    
        # Decode labels back to original class names
        true_label = label_encoder.inverse_transform([y_test_fold[0]])[0]
        predicted_label = label_encoder.inverse_transform([y_pred[0]])[0]

        # Store the OOF results
        row = {'True Label': true_label, 'Predicted Label': predicted_label}
        for class_idx, class_label in enumerate(class_names):
            row[f'Probability_{class_label}'] = y_proba[0, class_idx]
        
        oof_data.append(row)
        indices.append(X_test_fold.index[0])  # Store original index
    
    # Convert to DataFrame and set indices
    oof_df = pd.DataFrame(oof_data, index=indices)
    oof_df.index.name = 'Sample Index'

    # Calculate LOOCV accuracy
    loocv_accuracy = accuracy_score(oof_df['True Label'], oof_df['Predicted Label'])
    print(f"LOOCV Accuracy: {loocv_accuracy:.4f}")

    if save_figures_path:
        original_show = plt.show
        plt.show = lambda: None

        # Plot ROC Curve using real class names
        plot_roc_curve(oof_df['True Label'], oof_df.iloc[:, 2:].values, y, target_name, class_names, save_folder)
        plt.savefig(save_figures_path+"/"+os.path.basename(save_figures_path)+"_cancer_roc.png")
        plt.clf()

        # Plot Confusion Matrix using real class names
        plot_confusion_matrix(oof_df['True Label'], oof_df['Predicted Label'], target_name, class_names)
        plt.savefig(save_figures_path+"/"+os.path.basename(save_figures_path)+"_cancer_confusion_matrix.png")
        plt.clf()

        plot_separate_normalized_confusion_matrix(oof_df['True Label'], oof_df['Predicted Label'], target_name, class_names)
        plt.savefig(save_figures_path+"/"+os.path.basename(save_figures_path)+"_cancer_confusion_matrix_Normalized.png")
        plt.clf()

        plt.show = original_show

        if output_file:
            if len(np.unique(oof_df['True Label'])) == 2:  # Binary classification
                average_type = 'binary'
                pos_label = label_encoder.inverse_transform([1])[0]  # Get the actual positive class label
                probabilities_positive_class = oof_df[f'Probability_{pos_label}']
                auc = roc_auc_score(oof_df['True Label'], probabilities_positive_class)
            else:  # Multi-class classification
                average_type = 'macro'
                pos_label = None
                auc = roc_auc_score(oof_df['True Label'], oof_df.iloc[:, 2:].values, multi_class='ovr')

            precision = precision_score(oof_df['True Label'], oof_df['Predicted Label'], average=average_type, pos_label=pos_label)
            recall = recall_score(oof_df['True Label'], oof_df['Predicted Label'], average=average_type, pos_label=pos_label)
            f1 = f1_score(oof_df['True Label'], oof_df['Predicted Label'], average=average_type, pos_label=pos_label)
            balanced_acc = balanced_accuracy_score(oof_df['True Label'], oof_df['Predicted Label'])
            mcc = matthews_corrcoef(oof_df['True Label'], oof_df['Predicted Label'])

            save_metrics_to_file(os.path.basename(save_figures_path), precision, recall, f1, auc, balanced_acc, mcc, output_file)

    else:
        plot_roc_curve(oof_df['True Label'], oof_df.iloc[:, 2:].values, y, target_name, class_names, save_folder)
        plot_confusion_matrix(oof_df['True Label'], oof_df['Predicted Label'], target_name, class_names)

    # Train final model on the entire dataset
    model.fit(X, y_encoded)  # Train on numerical labels

    return model, loocv_accuracy, oof_df



def save_metrics_to_file(model_name, precision, recall, f1, auc, balanced_acc, mcc, output_file):
    # Check if the file exists
    file_exists = os.path.isfile(output_file)
    
    # Open the file in append mode
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header only if the file is empty
        if os.stat(output_file).st_size == 0:  # Check if the file is empty
            writer.writerow(['model', 'precision', 'recall', 'f1_score', 'auc', 'balanced_accuracy', 'mcc'])
        
        # Write the metrics for the current run
        writer.writerow([model_name, precision, recall, f1, auc, balanced_acc, mcc])



def train_model(train_df, test_df, target_name='target', model=None, param_grid=None, cv=5, search_method='grid', scoring=None, save_auc=None, save_figures_path=None, save_folder=None):
    """
    Trains a model with optional hyperparameter tuning using stratified cross-validation and finds the best decision threshold for each class.
    """
    average_auc = -1
    # If no model is provided, use RandomForestClassifier as the default
    if model is None:
        model = RandomForestClassifier(random_state=0, class_weight='balanced')

    # Splitting train and test data into features and target
    X_train = train_df.drop(columns=[target_name])
    y_train = train_df[target_name]
    
    X_test = test_df.drop(columns=[target_name])
    y_test = test_df[target_name]

    # Extract the class labels from y_train
    classes = np.unique(y_train)  # Automatically deduce class names from y_train

    # Configure the scoring metric based on user input
    if scoring == 'roc_auc_macro':
        scoring_metric = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='macro')
    elif scoring == 'roc_auc_micro':
        scoring_metric = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='micro')
    elif scoring == 'balanced_accuracy':
        scoring_metric = 'balanced_accuracy'
    else:
        # Default to accuracy if no or unrecognized scoring is provided
        scoring_metric = scoring or 'accuracy'

    # Set up stratified cross-validation
    stratified_cv = StratifiedKFold(n_splits=cv)

    # If hyperparameter tuning is requested
    if param_grid is not None:
        if search_method == 'grid':
            search = GridSearchCV(model, param_grid, cv=stratified_cv, scoring=scoring_metric, n_jobs=-1)
        elif search_method == 'random':
            search = RandomizedSearchCV(model, param_grid, cv=stratified_cv, scoring=scoring_metric, n_jobs=-1, random_state=0)
        
        # Perform the search and find the best model
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print(f"Best hyperparameters found: {search.best_params_}")
    else:
        # Fit the model without tuning
        model.fit(X_train, y_train)

    # Use stratified cross-validation to predict probabilities on the training data
    y_proba_cv = cross_val_predict(model, X_train, y_train, cv=stratified_cv, method='predict_proba')

    # Find best thresholds using cross-validation probabilities
    # best_thresholds = find_best_thresholds_cv(y_proba_cv, y_train, num_classes=len(classes))

    # Compute cross-validated accuracy
    cv_scores = cross_val_score(model, X_train, y_train, cv=stratified_cv, scoring='accuracy')
    cv_accuracy = cv_scores.mean()
    print(f"Cross-validated Accuracy: {cv_accuracy:.4f}")

    # Train final model on all training data
    model.fit(X_train, y_train)

    # Predict on test data
    y_proba_test = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # Plot classification results with the converted labels
    plot_classification_results(model, y_pred, y_proba_test, y_test, y_train, target_name,save_folder=save_auc)

    if save_figures_path:
        # Temporarily suppress plt.show()
        original_show = plt.show
        plt.show = lambda: None
         # Plot ROC Curve using existing function
        average_auc = plot_roc_curve(y_test, y_proba_test, y_train, target_name, model.classes_, save_folder)
        plt.savefig(save_figures_path+"/"+os.path.basename(save_figures_path)+"_cancer_roc.png")  # Save the loss vs epochs plot
        plt.clf()  # Clear the plot

        # Plot Confusion Matrix using existing function
        plot_confusion_matrix(y_test, y_pred, target_name, model.classes_)

        plt.savefig(save_figures_path+"/cancer_confusion_matrix.png")  # Save the confusion matrix plot
        plt.clf()  # Clear the plot


        plot_separate_normalized_confusion_matrix(y_test, y_pred, target_name, model.classes_)

        plt.savefig(save_figures_path+"/cancer_confusion_matrix_Normalized.png")  # Save the confusion matrix plot
        plt.clf()  # Clear the plot

        # Restore plt.show
        plt.show = original_show







  

    


       


   


    return model, average_auc

def train_model_withScaling(train_df, test_df, target_name='target', model=None, param_grid=None, cv=5, search_method='grid', scoring=None, save_auc=None, save_figures_path=None, save_folder=None):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df.drop(columns=[target_name]))
    X_test = scaler.transform(test_df.drop(columns=[target_name]))
    
    # Keep y_train and y_test unchanged
    y_train = train_df[target_name]
    y_test = test_df[target_name]

    model = LogisticRegression(penalty="l1", solver="saga", C=1/0.01, max_iter=5000, random_state=0)
    model.fit(X_train, y_train)

    model.scaler = scaler

   

   


    return model, -1

    

    
    

# def train_model_withScaling(train_df, test_df, target_name='target', model=None, param_grid=None, cv=5, search_method='grid', scoring=None, save_auc=None, save_figures_path=None, save_folder=None):
#     """
#     Trains a model with optional hyperparameter tuning using stratified cross-validation and finds the best decision threshold for each class.
#     """
#     average_auc = -1
#     # If no model is provided, use RandomForestClassifier as the default
#     if model is None:
#         model = RandomForestClassifier(random_state=0, class_weight='balanced')

#     # Initialize the scaler
#     scaler = StandardScaler()
    
#     # Fit on training data and transform both train and test
#     X_train = scaler.fit_transform(train_df.drop(columns=[target_name]))
#     X_test = scaler.transform(test_df.drop(columns=[target_name]))
    
#     # Keep y_train and y_test unchanged
#     y_train = train_df[target_name]
#     y_test = test_df[target_name]

#     # Extract the class labels from y_train
#     classes = np.unique(y_train)  # Automatically deduce class names from y_train

#     # Configure the scoring metric based on user input
#     if scoring == 'roc_auc_macro':
#         scoring_metric = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='macro')
#     elif scoring == 'roc_auc_micro':
#         scoring_metric = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='micro')
#     elif scoring == 'balanced_accuracy':
#         scoring_metric = 'balanced_accuracy'
#     else:
#         # Default to accuracy if no or unrecognized scoring is provided
#         scoring_metric = scoring or 'accuracy'

#     # Set up stratified cross-validation
#     stratified_cv = StratifiedKFold(n_splits=cv)

#     # If hyperparameter tuning is requested
#     if param_grid is not None:
#         if search_method == 'grid':
#             search = GridSearchCV(model, param_grid, cv=stratified_cv, scoring=scoring_metric, n_jobs=-1)
#         elif search_method == 'random':
#             search = RandomizedSearchCV(model, param_grid, cv=stratified_cv, scoring=scoring_metric, n_jobs=-1, random_state=0)
        
#         # Perform the search and find the best model
#         search.fit(X_train, y_train)
#         model = search.best_estimator_
#         print(f"Best hyperparameters found: {search.best_params_}")
#     else:
#         # Fit the model without tuning
#         model.fit(X_train, y_train)

#     # Use stratified cross-validation to predict probabilities on the training data
#     y_proba_cv = cross_val_predict(model, X_train, y_train, cv=stratified_cv, method='predict_proba')

#     # Find best thresholds using cross-validation probabilities
#     # best_thresholds = find_best_thresholds_cv(y_proba_cv, y_train, num_classes=len(classes))

#     # Compute cross-validated accuracy
#     cv_scores = cross_val_score(model, X_train, y_train, cv=stratified_cv, scoring='accuracy')
#     cv_accuracy = cv_scores.mean()
#     print(f"Cross-validated Accuracy: {cv_accuracy:.4f}")

#     # Train final model on all training data
#     model.fit(X_train, y_train)

#     # Predict on test data
#     y_proba_test = model.predict_proba(X_test)
#     y_pred = model.predict(X_test)

#     # Plot classification results with the converted labels
#     plot_classification_results(model, y_pred, y_proba_test, y_test, y_train, target_name,save_folder=save_auc)

#     if save_figures_path:
#         # Temporarily suppress plt.show()
#         original_show = plt.show
#         plt.show = lambda: None
#          # Plot ROC Curve using existing function
#         average_auc = plot_roc_curve(y_test, y_proba_test, y_train, target_name, model.classes_, save_folder)
#         plt.savefig(save_figures_path+"/"+os.path.basename(save_figures_path)+"_cancer_roc.png")  # Save the loss vs epochs plot
#         plt.clf()  # Clear the plot

#         # Plot Confusion Matrix using existing function
#         plot_confusion_matrix(y_test, y_pred, target_name, model.classes_)

#         plt.savefig(save_figures_path+"/cancer_confusion_matrix.png")  # Save the confusion matrix plot
#         plt.clf()  # Clear the plot


#         plot_separate_normalized_confusion_matrix(y_test, y_pred, target_name, model.classes_)

#         plt.savefig(save_figures_path+"/cancer_confusion_matrix_Normalized.png")  # Save the confusion matrix plot
#         plt.clf()  # Clear the plot

#         # Restore plt.show
#         plt.show = original_show







  

    
#     model.scaler = scaler

   

   


#     return model, average_auc


import pandas as pd

def test_trained_model(model, new_df, target_name='target'):
    """
    Tests the trained model on a new dataset and returns a DataFrame with probabilities.

    Parameters:
    - model (object): The trained model with an attached scaler.
    - new_df (pd.DataFrame): The new dataset to predict on.
    - target_name (str): The name of the target column.

    Returns:
    - results_df (pd.DataFrame): DataFrame with original indices and class probabilities.
    """
    if not hasattr(model, 'scaler'):
        raise ValueError("The trained model does not have an attached scaler. Ensure you are using the correct trained model.")

    # Extract features and scale them using the stored scaler
    X_new = model.scaler.transform(new_df.drop(columns=[target_name], errors='ignore'))  # Ignore errors if target column is missing

    # Make predictions
    y_pred = model.predict(X_new)
    y_proba = model.predict_proba(X_new)

    # Convert probabilities into a DataFrame with the original index
    class_labels = model.classes_  # Get class names
    results_df = pd.DataFrame(y_proba, columns=[f"{cls}" for cls in class_labels], index=new_df.index)

    # Add predicted labels
    results_df["Predicted_Label"] = y_pred

    return results_df









def train_meta_classifier(base_models, train_dfs, test_dfs, target_name='target', meta_classifier=LogisticRegression(random_state=0)):
    """
    Train a meta-classifier using predictions from base models.

    Parameters:
    - base_models: list of trained base models.
    - train_dfs: list of DataFrames corresponding to training feature subsets.
    - test_dfs: list of DataFrames corresponding to test feature subsets.
    - target_name: str, name of the target column (default is 'target').
    - meta_classifier: sklearn classifier, the meta-classifier to train (default is LogisticRegression).

    Returns:
    - meta_classifier: The trained meta-classifier.
    - accuracy_meta: Accuracy of the meta-classifier on the test set.
    - classification_report_meta: Classification report of the meta-classifier.
    """
    
    # Step 1: Generate Predictions from Base Models
    train_preds = []
    test_preds = []
    
    for model, train_df, test_df in zip(base_models, train_dfs, test_dfs):
        train_pred = model.predict_proba(train_df.drop(columns=[target_name]))
        test_pred = model.predict_proba(test_df.drop(columns=[target_name]))
        
        train_preds.append(train_pred)
        test_preds.append(test_pred)
    
    # Combine predictions into a single feature set for the meta-classifier
    X_meta_train = np.hstack(train_preds)
    X_meta_test = np.hstack(test_preds)
    
    # Use the original target for training the meta-classifier
    y_meta_train = train_dfs[0][target_name]
    y_meta_test = test_dfs[0][target_name]
    
    # Step 2: Train the Meta-Classifier
    meta_classifier.fit(X_meta_train, y_meta_train)
    
    # Predict on test data
    y_proba_test = meta_classifier.predict_proba(X_meta_test)

    y_pred = meta_classifier.predict(X_meta_test)

   

  

    # Plot classification results with the converted labels
    plot_classification_results(model, y_pred, y_proba_test, y_meta_test, y_meta_train, target_name)
    
    return meta_classifier

def train_xgboost_meta_classifier(base_models, train_dfs, test_dfs, target_name='target'):
    """
    Train an XGBoost meta-classifier using predictions from base models.

    Parameters:
    - base_models: list of trained base models.
    - train_dfs: list of DataFrames corresponding to training feature subsets.
    - test_dfs: list of DataFrames corresponding to test feature subsets.
    - target_name: str, name of the target column (default is 'target').

    Returns:
    - meta_classifier: The trained XGBoost meta-classifier.
    - y_meta_test_pred_decoded: The predicted labels for the test set.
    - y_meta_test: The true labels for the test set.
    """
    
    # Step 1: Generate Predictions from Base Models
    train_preds = []
    test_preds = []

    for model, train_df, test_df in zip(base_models, train_dfs, test_dfs):
        train_pred = model.predict_proba(train_df.drop(columns=[target_name]))
        test_pred = model.predict_proba(test_df.drop(columns=[target_name]))

        train_preds.append(train_pred)
        test_preds.append(test_pred)

    # Combine predictions into a single feature set for the meta-classifier
    X_meta_train = np.hstack(train_preds)
    X_meta_test = np.hstack(test_preds)

    # Use the original target for training the meta-classifier
    y_meta_train = train_dfs[0][target_name]
    y_meta_test = test_dfs[0][target_name]

    # Encode the target labels as integers
    label_encoder = LabelEncoder()
    y_meta_train_encoded = label_encoder.fit_transform(y_meta_train)
    y_meta_test_encoded = label_encoder.transform(y_meta_test)

    # Step 2: Train the XGBoost Meta-Classifier
    meta_classifier = XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.1)
    meta_classifier.fit(X_meta_train, y_meta_train_encoded)

    # Step 3: Make predictions on the test set
    y_meta_test_pred_encoded = meta_classifier.predict(X_meta_test)

    # Decode the predictions back to original labels
    y_meta_test_pred_decoded = label_encoder.inverse_transform(y_meta_test_pred_encoded)
    
    return meta_classifier, y_meta_test_pred_decoded, y_meta_test
