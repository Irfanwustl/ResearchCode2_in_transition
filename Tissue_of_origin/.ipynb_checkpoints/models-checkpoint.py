from plotting_utils import plot_classification_results, plot_roc_curve, plot_confusion_matrix
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


from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score






def combine_train_test(train_df, test_df):
    """
    Combines train and test datasets into a single dataset for unified processing.

    Parameters:
        train_df (DataFrame): Training data with features and target column.
        test_df (DataFrame): Test data with features and target column.

    Returns:
        combined_df (DataFrame): The combined dataset with train and test data.
    """
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    return combined_df

def train_model_loocv(dataset, target_name='target', model=None, param_grid=None, search_method='grid', scoring=None, save_folder=None):
    """
    Trains a model using Leave-One-Out Cross-Validation (LOOCV) on a single dataset, plots the ROC curve, and confusion matrix.

    Parameters:
        dataset (DataFrame): Combined dataset with features and target column.
        target_name (str): The name of the target column.
        model (object): Machine learning model to train. Defaults to RandomForestClassifier.
        param_grid (dict): Hyperparameter grid for tuning. If None, no tuning is performed.
        search_method (str): Method for hyperparameter search ('grid' or 'random').
        scoring (str): Metric for model evaluation (e.g., 'roc_auc_macro', 'accuracy').
        save_folder (str): Folder to save ROC data files. If None, files are not saved.

    Returns:
        model (object): The trained model.
        loocv_accuracy (float): The accuracy from LOOCV.
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

    # Set up Leave-One-Out Cross-Validation
    loo = LeaveOneOut()

    # If hyperparameter tuning is requested
    if param_grid is not None:
        if search_method == 'grid':
            search = GridSearchCV(model, param_grid, cv=loo, scoring=scoring_metric, n_jobs=-1)
        elif search_method == 'random':
            search = RandomizedSearchCV(model, param_grid, cv=loo, scoring=scoring_metric, n_jobs=-1, random_state=0)

        # Perform the search and find the best model
        search.fit(X, y)
        model = search.best_estimator_
        print(f"Best hyperparameters found: {search.best_params_}")
    
    # Perform LOOCV
    predictions = []
    probabilities = []
    true_labels = []

    for train_index, test_index in loo.split(X):
        # Split data into training and testing folds for LOOCV
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        # Train the model on the training fold
        model.fit(X_train_fold, y_train_fold)

        # Predict on the test fold
        y_pred = model.predict(X_test_fold)
        y_proba = model.predict_proba(X_test_fold)

        predictions.append(y_pred[0])  # Append the predicted label
        probabilities.append(y_proba[0])  # Append the predicted probabilities
        true_labels.append(y_test_fold.iloc[0])  # Append the true label

    # Convert lists to numpy arrays
    probabilities = np.array(probabilities)
    true_labels = np.array(true_labels)

    # Calculate LOOCV accuracy
    loocv_accuracy = accuracy_score(true_labels, predictions)
    print(f"LOOCV Accuracy: {loocv_accuracy:.4f}")

    # Train final model on the entire dataset
    model.fit(X, y)

    # Plot ROC Curve using existing function
    plot_roc_curve(pd.Series(true_labels), probabilities, y, target_name, classes, save_folder)

    # Plot Confusion Matrix using existing function
    plot_confusion_matrix(true_labels, predictions, target_name, classes)
   

    return model, loocv_accuracy


def train_model(train_df, test_df, target_name='target', model=None, param_grid=None, cv=5, search_method='grid', scoring=None, save_auc=None):
    """
    Trains a model with optional hyperparameter tuning using stratified cross-validation and finds the best decision threshold for each class.
    """
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

    return model










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
