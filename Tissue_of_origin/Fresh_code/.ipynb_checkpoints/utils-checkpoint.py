import re
import numpy as np
import models
import plots

import pandas as pd

def merge_and_clean_labels(dataframes):
    """
    Merges a list of DataFrames along columns, keeps only one 'True Label' column, 
    and renames it to 'cohort'.

    Parameters:
    dataframes (list of pd.DataFrame): List of DataFrames to be concatenated.

    Returns:
    pd.DataFrame: A cleaned DataFrame with a single 'cohort' column instead of multiple 'True Label' columns.
    """
    # Concatenate all DataFrames along columns
    merged_df = pd.concat(dataframes, axis=1)

    # Find all columns that end with 'True Label'
    true_label_columns = [col for col in merged_df.columns if col.endswith('True Label')]

    # Drop all but the first 'True Label' column
    merged_df.drop(columns=true_label_columns[1:], inplace=True)
    # Rename the first 'True Label' column to 'cohort'
    merged_df.rename(columns={true_label_columns[0]: 'cohort'}, inplace=True)

    return merged_df

# Function to determine 'True Label'
def assign_label(sample_id):
    if sample_id.startswith(('W', 'B', 'b')):
        return 'Bladder'
    elif sample_id.startswith(('R', 'r')):
        return 'RCC'
    elif sample_id.startswith(('P', 'p')):
        return 'Prostate'
    else:
        return 'Healthy'

def format_mix_probs(df1, df2):
    # Convert index to a column for merging
    df1 = df1.reset_index().rename(columns={"index": "library"})
    df2 = df2.reset_index().rename(columns={"index": "library"})

    # Merge df1 with df2 to get the 'cohort' column
    result = df1.merge(df2[['library', 'cohort']], on="library", how="left")

    # Map cohort to probability columns
    result["cohort_prob"] = result.apply(
        lambda row: row[row["cohort"]] if row["cohort"] in result.columns else np.nan, axis=1
    )

    # Compute pct_cancer based on conditions
    def compute_pct_cancer(library):
        if "UC1" in library:
            return 100
        parts = library.split("_")
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else np.nan

    result["pct_cancer"] = result["library"].apply(compute_pct_cancer)

    # Set 'library' as index, keep only relevant columns, and sort
    result = result.set_index("library")[["cohort", "pct_cancer", "cohort_prob"]]
    result = result.sort_values(by=["cohort", "pct_cancer"])

    return result

def format_mix2_probs(df, tf):
    # Reset index to move row names to a column
    df = df.reset_index().rename(columns={'library': 'mix_name'})
    
    # Rename 'Bladder' column to 'bladder_prob'
    df = df.rename(columns={'Bladder': 'bladder_prob'})
    
    # Perform left join with tf on 'mix_name'
    df = df.merge(tf, on='mix_name', how='left')
    
    # Set 'mix_name' back as index
    df = df.set_index('mix_name')
    
    # Select required columns and sort by 'orig_lib' and 'est_tf'
    df = df[['orig_lib', 'est_tf', 'tumor_fraction', 'bladder_prob']].sort_values(by=['orig_lib', 'est_tf'])
    
    return df

def LOD(model, mixture1, mixture2, tf, figure_path=None):
    predicted1 = LOD_analysis(model, mixture1, figure_path)
    predicted2 = LOD_analysis2(model, mixture2, tf, figure_path)
    return predicted1, predicted2
    
    
    

def LOD_analysis2(model, mixture, tf, figure_path=None):
    predicted,_ = models.predict_model(model, mixture)
    formatted = format_mix2_probs(predicted, tf)
    plots.plot_LOD2(formatted, figure_path)
    return predicted
   

def LOD_analysis(model, mixture, figure_path=None):
    predicted,_ = models.predict_model(model, mixture)
    formatted = format_mix_probs(predicted, mixture)
    plots.plot_LOD(formatted, figure_path)
    return predicted

