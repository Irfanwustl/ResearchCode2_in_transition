from plotting_utils import plot_classification_results
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
import numpy as np

def random_forest_model(train_df, test_df, target_name='target', n_estimators=1000, random_state=0):
    # Splitting train and test data into features and target
    X_train = train_df.drop(columns=[target_name])
    y_train = train_df[target_name]
    
    X_test = test_df.drop(columns=[target_name])
    y_test = test_df[target_name]
    
    # Initialize and fit the RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    
    # Plot classification results
    plot_classification_results(rf_model, X_test, y_test, y_train, target_name)
    
    return rf_model


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
    
    # Plot classification results
    plot_classification_results(meta_classifier, X_meta_test, y_meta_test, y_meta_train, target_name)
    
    return meta_classifier

