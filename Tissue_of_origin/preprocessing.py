from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd


def combine_feature_dfs_with_target(dfs_with_prefixes, target_column='target'):
    """
    Combine multiple DataFrames with a common target column, ensuring indices and target values match.
    Adds a prefix to each feature set for identification.

    Parameters:
    dfs_with_prefixes : list of tuples
        A list of tuples where each tuple contains a DataFrame and its corresponding prefix.
        Example: [(df1, 'CNA_'), (df2, 'Ratio_'), (df3, 'EndMotif_')]
    target_column : str, optional (default='target')
        Name of the target column that is common across all DataFrames.

    Returns:
    combined_df : DataFrame
        A single DataFrame with all features combined, prefixed, and the target column appended.
    """
    # Ensure there is at least one DataFrame
    if not dfs_with_prefixes:
        raise ValueError("At least one DataFrame must be provided.")

    combined_features = pd.DataFrame()

    for df, prefix in dfs_with_prefixes:
        # Check that all DataFrames have the same index
        if not dfs_with_prefixes[0][0].index.equals(df.index):
            raise ValueError("Indices do not match across the DataFrames.")

        # Check that all DataFrames have the same target column values
        if not dfs_with_prefixes[0][0][target_column].equals(df[target_column]):
            raise ValueError(f"Target columns do not match across the DataFrames.")

        # Drop the target column and add prefix to the feature columns
        df_prefixed = df.drop(columns=[target_column]).add_prefix(prefix)

        # Concatenate to the combined DataFrame
        combined_features = pd.concat([combined_features, df_prefixed], axis=1)

    # Reattach the target column and preserve the index
    combined_features[target_column] = dfs_with_prefixes[0][0][target_column]
    combined_features.index = dfs_with_prefixes[0][0].index

    # Convert all column names to strings, replacing tuples with hyphen-separated strings
    combined_features.columns = ['-'.join(col) if isinstance(col, tuple) else col for col in combined_features.columns]

    return combined_features

def standardize_dataframe(df, target_column='target'):
    """
    Standardizes the features of a DataFrame, excluding the target column.

    Parameters:
    df (pd.DataFrame): The input DataFrame with features and target.
    target_column (str): The name of the target column to be excluded from standardization.
                         Default is 'target'.

    Returns:
    pd.DataFrame: A DataFrame with standardized features and the original target column.
    """
    # Separate the features and target
    features = df.drop(columns=[target_column])
    target = df[target_column]

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit and transform the features
    scaled_features = scaler.fit_transform(features)

    # Convert the scaled features back to a DataFrame, keeping the original index
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns, index=df.index)

    # Add the target column back to the DataFrame
    scaled_df[target_column] = target

    return scaled_df



def preprocess_dataframe(df):
    # Transpose the dataframe
    df_transposed = df.transpose()
    
    # Extract target labels from the sample names (which are now in the index)
    df_transposed['target'] = df_transposed.index.map(get_label)


    
    return df_transposed

def get_label(sample_id):
    if sample_id.startswith('W') or sample_id.startswith('B'):
        return 'Bladder'
    elif sample_id.startswith('R'):
        return 'RCC'
    elif sample_id.startswith('P'):
        return 'Prostate'
    else:
        return 'Healthy'



def remove_nan_inf_columns(df):
    # Replace inf and -inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop columns with any NaN values
    df_cleaned = df.dropna(axis=1, how='any')
    
    return df_cleaned


def make_value_unique(df, target_value, epsilon=1e-6):
    """
    Modifies the DataFrame in place by adding small random values to instances of a specified target value 
    in all columns, ensuring that the target value becomes unique in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to be modified.
    target_value (numeric): The value in the DataFrame that needs to be made unique.
    epsilon (float, optional): The range within which random values are generated. 
                               Default is 1e-6.

    Returns:
    pd.DataFrame: The modified DataFrame with unique values in place of the target value.
    """
    
    # Iterate over each column in the DataFrame
    for col in df.columns:
        # Create a boolean mask identifying the rows where the value in the current column equals target_value
        mask = df[col] == target_value
        
        # Generate small random values between epsilon and 2*epsilon
        # These values will be added to the target_value to make it unique
        random_values = np.random.uniform(low=epsilon, high=2*epsilon, size=mask.sum())
        
        # Add the generated random values to the cells that match the target_value
        df.loc[mask, col] = df.loc[mask, col] + random_values
    
    # Return the modified DataFrame with unique values replacing the target_value
    return df


def plot_class_distribution(df, target_name='target'):
    # Count the frequency of each class in the target column
    class_counts = df[target_name].value_counts()

    # Plot the class distribution
    plt.figure(figsize=(10, 6))
    ax = class_counts.plot(kind='bar', color='skyblue')
    
    # Add number on top of each bar
    for p in ax.patches:
        ax.annotate(str(int(p.get_height())), (p.get_x() * 1.005, p.get_height() * 1.005))
    
    plt.title(f'Class Distribution in {target_name}')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)
    plt.show()

def stratified_train_test_split(df, test_size=0.2, random_state=0):
    # Assuming the target column is named 'Target'
    X = df.drop(columns=['target'])  # Features
    y = df['target']  # Target
    
    # Perform stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Combine X and y back into DataFrames for train and test sets
    train_df = X_train.copy()
    train_df['target'] = y_train
    
    test_df = X_test.copy()
    test_df['target'] = y_test
    
    return train_df, test_df



def undersample_majority_class(df, target_name='target', random_state=0):
    # Split the data into features and target
    X = df.drop(columns=[target_name])
    y = df[target_name]
    
    # Define the RandomUnderSampler with a specified random state for reproducibility
    rus = RandomUnderSampler(random_state=random_state)
    
    # Perform undersampling
    X_resampled, y_resampled = rus.fit_resample(X, y)
    
    # Combine the resampled features and target back into a DataFrame
    df_resampled = X_resampled.copy()
    df_resampled[target_name] = y_resampled
    
    return df_resampled


def normalize_features_by_sample(df, target_column='target'):
    """
    Normalizes each feature in the DataFrame by the total counts across samples 
    and scales the values to 1e6, leaving the target column unaffected. This is a CPM like normalization.
    
    Parameters:
    df (pd.DataFrame): The DataFrame where samples are in rows and features are in columns.
    target_column (str): The name of the column that should remain unaffected by normalization.
    
    Returns:
    pd.DataFrame: The normalized DataFrame with the target column unchanged.
    """
    # Separate the target column from the features
    features = df.drop(columns=[target_column])
    target = df[target_column]
    
    # Sum across rows (samples) for each feature (column-wise sum)
    total_counts = features.sum(axis=1)
    
    # Divide each element by the corresponding sample's total counts and scale by 1e6
    normalized_features = features.divide(total_counts, axis=0) * 1e6
    
    # Combine the normalized features with the target column
    normalized_df = normalized_features.copy()
    normalized_df[target_column] = target
    
    return normalized_df


def check_scaling(df, target_column='target'):
    """
    Checks the mean and standard deviation of each feature in the DataFrame,
    excluding the target column.

    Parameters:
    df (pd.DataFrame): The input DataFrame with features and target.
    target_column (str): The name of the target column to be excluded from the checks.
                         Default is 'target'.

    Returns:
    pd.Series: A Series of means for each feature.
    pd.Series: A Series of standard deviations for each feature.
    """
    # Drop the target column to get only features
    features_df = df.drop(columns=[target_column])
    
    # Calculate the mean and standard deviation of each feature
    means = features_df.mean()
    stds = features_df.std()

    # Print the results
    print("Means:\n", means)
    print("Standard Deviations:\n", stds)

    return means, stds


# Function to manually filter columns based on a prefix
def filter_columns_by_prefix(df, prefix):
    def column_matches(col):
        if isinstance(col, tuple):
            return col[0].startswith(prefix)
        elif isinstance(col, str):
            return col.startswith(prefix)
        return False

    # Apply the filtering logic
    return df[[col for col in df.columns if column_matches(col) or col == 'target']]


def extract_feature_importances(model, df, target_name='target'):
    """
    Extracts and returns the feature importances from a trained model.

    Parameters:
    - model: The trained model (must have feature_importances_ attribute).
    - df: DataFrame containing the data (excluding the target column).
    - target_name: Name of the target column in df.

    Returns:
    - A DataFrame sorted by feature importance.
    """
    # Check if the model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
    else:
        raise ValueError("The provided model does not have the 'feature_importances_' attribute.")

    # Create a DataFrame for feature importances
    importances_df = pd.DataFrame({
        'Feature': df.drop(columns=[target_name]).columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    return importances_df


def subset_top_k_features(train_df, test_df, k, feature_importances_dict, target_name='target'):
    """
    Subsets the train and test DataFrames with the top k features from each feature importance DataFrame.
    
    Parameters:
    - train_df: DataFrame containing the training data.
    - test_df: DataFrame containing the testing data.
    - k: The number of top features to select from each feature importance DataFrame.
    - feature_importances_dict: Dictionary with feature importance DataFrames. Keys should correspond to feature group names.
    - target_name: Name of the target column.
    
    Returns:
    - Subsetted train_df and test_df containing only the top k features from each feature group.
    """
    selected_features = set()

    # Iterate over each feature importance DataFrame and select top k features
    for feature_group, importance_df in feature_importances_dict.items():
        top_k_features = importance_df['Feature'].head(k)
        selected_features.update(top_k_features)

    # Convert the set to a list
    selected_features = list(selected_features)

    # Subset the train and test DataFrames
    train_subset = train_df[selected_features + [target_name]]
    test_subset = test_df[selected_features + [target_name]]

    return train_subset, test_subset



