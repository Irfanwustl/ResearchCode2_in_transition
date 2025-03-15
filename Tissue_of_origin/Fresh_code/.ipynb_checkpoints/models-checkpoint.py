from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
import plots

from sklearn.model_selection import LeaveOneOut, GridSearchCV
import numpy as np

def train_model_loocv(train_data, test_data, response_col, model_class, param_grid, scoring="accuracy", cv_folds=5):
    """
    Trains a machine learning model using Leave-One-Out Cross-Validation (LOOCV).
    - First, finds the best hyperparameters using k-fold CV **on train_data only**.
    - Then applies LOOCV on **the combined (train + test) data**.

    Parameters:
    - train_data: DataFrame containing training features and target column.
    - test_data: DataFrame containing test features and target column.
    - response_col: Name of the target column.
    - model_class: A machine learning model class (e.g., LogisticRegression).
    - param_grid: List of dictionaries containing hyperparameter search space.
    - scoring: Scoring metric for model selection (default: "accuracy").
    - cv_folds: Number of folds for hyperparameter tuning (default: 5).

    Returns:
    - A final trained pipeline trained on the **full dataset (train + test)** using best hyperparameters.
    - The final LOOCV AUC score.
    """

    # **Step 1: Hyperparameter Tuning Using 5-Fold CV on Train Data Only**
    print("\n🔍 Performing Grid Search with k-Fold CV to find best hyperparameters (on train data only)...")

    # Extract train features & target
    X_train = train_data.drop(columns=[response_col])
    y_train = train_data[response_col].astype("category")

    # Define pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model_class())  # Placeholder model
    ])

    # Run Grid Search CV on Train Data
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid={"model__" + key: value for param_set in param_grid for key, value in param_set.items()},
        cv=cv_folds,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("\n✅ Best hyperparameters found:", best_params)

    # **Step 2: Run LOOCV on the Combined (Train + Test) Data**
    print("\n🎯 Running LOOCV with the best hyperparameters on (train + test) data...")

    # Combine Train & Test Data
    full_data = pd.concat([train_data, test_data])
    X_full = full_data.drop(columns=[response_col])
    y_full = full_data[response_col].astype("category")

    loo = LeaveOneOut()

    all_preds = []  # Store all predicted probabilities
    all_y_test = []  # Store all actual test labels
    #all_pred_probs = []  # ✅ Store all predicted probabilities for AUC calculation


    for train_idx, test_idx in loo.split(X_full):
        X_train_loo, X_test_loo = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train_loo, y_test_loo = y_full.iloc[train_idx], y_full.iloc[test_idx]

        # ✅ Use a Pipeline to ensure StandardScaler is included
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model_class(**{key.replace("model__", ""): value for key, value in best_params.items()}))
        ])
        pipeline.fit(X_train_loo, y_train_loo)

        # Predict probabilities for the left-out sample
        y_pred_proba = pipeline.predict_proba(X_test_loo)

        # ✅ Store predictions correctly
        all_preds.append(y_pred_proba[0])  # Append as a NumPy array
        all_y_test.append(str(y_test_loo.iloc[0]))  # Store actual label as a string to avoid type mismatch
      
    

   
    # ✅ Convert stored predictions to arrays
   
    all_preds = np.vstack(all_preds)  # Stack into a single NumPy array
  
    # ✅ Ensure class labels come from `y_full` for consistency
    
    class_labels = pipeline.named_steps["model"].classes_
 

    
    all_pred_probs = all_preds
    all_preds = all_preds.argmax(axis=1)  # Get the index of the highest probability

  
    # ✅ Convert stored predictions to arrays

    
    # ✅ Map indices back to class names
    all_preds = np.array([class_labels[idx] for idx in all_preds])
    
    
    # ✅ Ensure `all_y_test` is also a NumPy array of strings
    all_y_test = np.array(all_y_test).astype(str)


    # ✅ Compute overall AUC and plot Confusion Matrix + ROC
    plots.plot_confusion_matrix(all_y_test, all_preds, target_name=response_col, classes=class_labels)
    average_auc = plots.plot_roc_curve( pd.Series(all_y_test), all_pred_probs, y_full, target_name=response_col, classes=class_labels)

    print(f"\n📊 Final LOOCV AUC Score: {average_auc:.4f}")

    # **Step 3: Train Final Model on the Full (Train + Test) Data**
    print("\n🚀 Training final model on the full dataset with best hyperparameters...")
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),  # Use StandardScaler inside the pipeline
        ("model", model_class(**{key.replace("model__", ""): value for key, value in best_params.items()}))
    ])
    final_pipeline.fit(X_full, y_full)  # Train on full dataset

    return final_pipeline, average_auc  # ✅ Returning the final trained model


def train_model(train_data, response_col, model_class, param_grid, cv=5, scoring='accuracy', n_jobs=-1):
    """
    Trains a machine learning model using cross-validation and selects the best model.

    Parameters:
    - train_data: DataFrame containing features and target column.
    - response_col: Name of the target column.
    - model_class: A machine learning model class (e.g., LogisticRegression).
    - param_grid: List of dictionaries containing hyperparameter search space.
    - cv: Number of cross-validation folds.
    - scoring: Scoring metric for model selection.
    - n_jobs: Number of parallel jobs for GridSearchCV (-1 uses all available cores).

    Returns:
    - A trained pipeline with the best model.
    """
    # Separate features (X) and target (y)
    X = train_data.drop(columns=[response_col])
    y = train_data[response_col].astype("category")

    # Standardization
    preprocessor = StandardScaler()

    # Define the pipeline with a placeholder model
    pipeline = Pipeline([
        ("scaler", preprocessor),
        ("model", model_class())  # Initialize model without parameters
    ])

    # Define GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid={"model__" + key: value for param_set in param_grid for key, value in param_set.items()},
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1
    )
    
    # Train the model using cross-validation
    grid_search.fit(X, y)

    # Print the best hyperparameters within the function
    print("\nBest hyperparameters found:", grid_search.best_params_)

    # Return the trained pipeline with the best model (ensuring it contains the fitted scaler)
    return grid_search.best_estimator_


def predict_model(model, data, target_column="cohort", plot_cm=False, plot_roc=False,save_folder=None):
    """
    Predicts the output for the given dataset using the provided trained model.

    Parameters:
    - model: The trained pipeline model.
    - data: DataFrame containing the features and target column.
    - target_column: The name of the column to be excluded from features (default: "cohort").

    Returns:
    - DataFrame containing the predicted probabilities with class labels as column names.
    """
    # Ensure the input data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    # Exclude the target column if it exists
    feature_data = data.drop(columns=[target_column], errors="ignore")

    # Ensure features match those from training
    trained_features = model.named_steps["scaler"].feature_names_in_
    feature_data = feature_data[trained_features]  # Ensures correct feature order

    # Ensure model supports probability prediction
    if not hasattr(model.named_steps["model"], "predict_proba"):
        raise ValueError("The provided model does not support probability prediction.")

    # ✅ Best Practice: Use `model.predict_proba()` to ensure preprocessing is applied
    pred_probs = model.predict_proba(feature_data)

    # ❗ Alternative: If using `named_steps["model"].predict_proba()`, manually apply scaler first
    # feature_data_scaled = model.named_steps["scaler"].transform(feature_data)
    # pred_probs = model.named_steps["model"].predict_proba(feature_data_scaled)

    # Get class labels from the model
    class_labels = model.named_steps["model"].classes_

    # Convert to DataFrame with proper row and column labels
    pred_df = pd.DataFrame(pred_probs, columns=class_labels, index=feature_data.index)

    # Convert probabilities to class predictions
    y_pred = pred_df.idxmax(axis=1)  # Selects the class with the highest probability

    # Extract true labels
    if target_column in data.columns:
        y_test = data[target_column]
    else:
        y_test = None

    # Plot confusion matrix if ground truth labels are available
    if plot_cm and y_test is not None:
        plots.plot_confusion_matrix(y_test, y_pred, target_name=target_column, classes=class_labels)
        plots.plot_separate_normalized_confusion_matrix(y_test, y_pred, target_name=target_column, classes=class_labels)

    # Plot ROC curve if enabled
    average_auc = None
    if plot_roc and y_test is not None:
        average_auc = plots.plot_roc_curve(y_test, pred_probs, data[target_column], target_name=target_column, classes=class_labels, save_folder=save_folder)


    return pred_df, average_auc  # Return predicted probabilities as before
