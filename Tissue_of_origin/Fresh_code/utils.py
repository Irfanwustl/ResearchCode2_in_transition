import re
import numpy as np
import models
import plots

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

def format_mix2_probs(df, tf_mix2):
    # Reset index to move row names to a column
    df = df.reset_index().rename(columns={'library': 'mix_name'})
    
    # Rename 'Bladder' column to 'bladder_prob'
    df = df.rename(columns={'Bladder': 'bladder_prob'})
    
    # Perform left join with tf_mix2 on 'mix_name'
    df = df.merge(tf_mix2, on='mix_name', how='left')
    
    # Set 'mix_name' back as index
    df = df.set_index('mix_name')
    
    # Select required columns and sort by 'orig_lib' and 'est_tf'
    df = df[['orig_lib', 'est_tf', 'tumor_fraction', 'bladder_prob']].sort_values(by=['orig_lib', 'est_tf'])
    
    return df

def LOD_analysis2(model, mixture, tf_mix2):
    predicted,_ = models.predict_model(model, mixture)
    formatted = format_mix2_probs(predicted, tf_mix2)
    plots.plot_LOD2(formatted)
   

def LOD_analysis(model, mixture):
    predicted,_ = models.predict_model(model, mixture)
    formatted = format_mix_probs(predicted, mixture)
    plots.plot_LOD(formatted)

