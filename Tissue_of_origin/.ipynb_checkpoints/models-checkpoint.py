from plotting_utils import plot_classification_results
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


def train_model(train_df, test_df, target_name='target', model=None, param_grid=None, cv=5, search_method='grid', scoring=None):
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
    #best_thresholds = find_best_thresholds_cv(y_proba_cv, y_train, num_classes=len(classes))

    # Train final model on all training data
    model.fit(X_train, y_train)

    # Predict on test data
    y_proba_test = model.predict_proba(X_test)

    y_pred = model.predict(X_test)

   

  

    # Plot classification results with the converted labels
    plot_classification_results(model, y_pred, y_proba_test, y_test, y_train, target_name)

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
