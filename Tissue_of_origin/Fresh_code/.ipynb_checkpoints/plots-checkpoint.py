import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.preprocessing import label_binarize
import pandas as pd
import os



def plot_LOD(data, figure_path):
    """
    Plots the End Motif Model's Limit of Detection by Cohort.

    Parameters:
    - data: DataFrame containing 'pct_cancer', 'cohort_prob', and 'cohort' columns.

    Returns:
    - None (Displays the plot)
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x="pct_cancer", y="cohort_prob", hue="cohort", style="cohort", s=100)
    sns.lineplot(data=data, x="pct_cancer", y="cohort_prob", hue="cohort", legend=False)

    # Reverse the X-axis
    plt.gca().invert_xaxis()

    # Set labels and title
    #plt.title("End Motif Model\nLimit of Detection by Cohort", fontsize=14)
    plt.xlabel("Percent of Fragments from Cancer Patient", fontsize=12)
    plt.ylabel("Predicted Probability", fontsize=12)

    # Format X-axis as percentage
    plt.xticks(ticks=[100, 75, 50, 25, 10, 5, 1], labels=[f"{x}%" for x in [100, 75, 50, 25, 10, 5, 1]])

    # Improve layout
    plt.legend(title="Cohort")
    plt.grid(True, linestyle="--", alpha=0.6)
    sns.despine()

    plt.savefig(figure_path+"/"+os.path.basename(figure_path)+"_LOD.png", dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

def plot_LOD2(data, figure_path):
    """
    Plots the relationship between estimated tumor fraction and predicted probability.

    Parameters:
        data (pd.DataFrame): A DataFrame containing 'est_tf', 'bladder_prob', and 'orig_lib' columns.

    Returns:
        None (Displays the plot)
    """
    plt.figure(figsize=(8, 6))
    
    # Scatter plot with lines for each original library group
    sns.scatterplot(data=data, x="est_tf", y="bladder_prob", hue="orig_lib", marker="o", edgecolor="black")
    sns.lineplot(data=data, x="est_tf", y="bladder_prob", hue="orig_lib", legend=False, alpha=0.7)

    # Customize labels
    plt.xlabel("Estimated Tumor Fraction")
    plt.ylabel("Predicted Probability")
    plt.title("Limit of Detection Plot")

    # Reverse x-axis scale
    plt.gca().invert_xaxis()
    plt.xticks(ticks=[0.12, 0.10, 0.08, 0.06, 0.04, 0.02])

    plt.legend(title="Original Library")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(figure_path+"/"+os.path.basename(figure_path)+"_LOD2.png", dpi=300, bbox_inches='tight')

    plt.show()

def plot_tumor_fraction(data):
    """
    Plots the relationship between intended tumor fraction (-est_tf) and measured tumor fraction.
    Includes a reference regression line (y = -x) and facets by 'orig_lib'.

    Parameters:
        data (pd.DataFrame): A DataFrame containing 'est_tf', 'tumor_fraction', and 'orig_lib' columns.

    Returns:
        None (Displays the plot)
    """
    # Create a modified DataFrame to store -est_tf
    data = data.copy()
    data['neg_est_tf'] = -data['est_tf']  # Create a new column with negated values

    # Scatter plot of tumor_fraction vs. -est_tf
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x="neg_est_tf", y="tumor_fraction")

    # Add reference line (y = -x)
    plt.axline((0, 0), slope=-1, linestyle="dashed", color="red")

    # Customize labels and ticks
    plt.xlabel("Intended Tumor Fraction")
    plt.ylabel("Measured Tumor Fraction")
    plt.xticks(ticks=[-0.02, -0.04, -0.06, -0.08, -0.10, -0.12], labels=["0.02", "0.04", "0.06", "0.08", "0.10", "0.12"])

    plt.show()

    # Facet by 'orig_lib'
    g = sns.FacetGrid(data, col="orig_lib", col_wrap=3, height=4)
    g.map_dataframe(sns.scatterplot, x="neg_est_tf", y="tumor_fraction")

    plt.show()


def plot_confusion_matrix(y_test, y_pred, target_name, classes):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))  # Create a side-by-side plot

    # Plot non-normalized confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax[0])
    ax[0].set_title(f'Confusion Matrix for {target_name}')
    
    # Plot normalized confusion matrix
    cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')
    disp_normalized = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=classes)
    disp_normalized.plot(cmap=plt.cm.Blues, values_format='.2f', ax=ax[1])
   # ax[1].set_title(f'Normalized Confusion Matrix for {target_name}')

    plt.tight_layout()
    plt.show()

def plot_separate_normalized_confusion_matrix(y_test, y_pred, target_name, classes):
    """Function to plot the normalized confusion matrix separately."""
    cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp_normalized = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=classes)
    disp_normalized.plot(cmap=plt.cm.Blues, values_format='.2f', ax=ax)
   # ax.set_title(f'Separate Normalized Confusion Matrix for {target_name}')
    plt.show()


def plot_roc_curve(y_test, y_pred_proba, y_train, target_name, classes, save_folder=None):
    plt.figure(figsize=(12, 10))
    
    if save_folder and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if len(np.unique(y_train)) == 2:  # Binary classification case
        # Dynamically determine the positive label
        label_counts = pd.Series(y_train).value_counts()
        pos_label = label_counts.index[-1]  # Treat the less frequent label as the positive class
    
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1], pos_label=pos_label)
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        plt.plot(fpr, tpr, label=f'{target_name} ROC curve (AUC = {roc_auc:.2f})')

        average_auc = roc_auc
    
        # Save data to file
        if save_folder:
            file_path = os.path.join(save_folder, f"{target_name}.tsv")
            with open(file_path, 'w') as file:
                file.write("FPR\tTPR\tAUC\n")
                for fp, tp in zip(fpr, tpr):
                    file.write(f"{fp:.6f}\t{tp:.6f}\t{roc_auc:.6f}\n")
    else:
        # Multiclass case
        y_test_bin = label_binarize(y_test, classes=classes)
        
        # Initialize AUCs
        average_auc = 0
        macro_auc = roc_auc_score(y_test_bin, y_pred_proba, average='macro')
        micro_auc = roc_auc_score(y_test_bin, y_pred_proba, average='micro')

        # Compute class-wise AUC and update the average AUC
        for i, class_label in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
            average_auc += roc_auc

            # Save data to separate files for each class
            if save_folder:
                file_path = os.path.join(save_folder, f"{class_label}.tsv")
                with open(file_path, 'w') as file:
                    file.write("FPR\tTPR\tAUC\n")
                    for fp, tp in zip(fpr, tpr):
                        file.write(f"{fp:.6f}\t{tp:.6f}\t{roc_auc:.6f}\n")
        
        average_auc /= len(classes)  # Calculate the average AUC

        # Initialize the legend text
        legend_text = (
            # f'Macro AUC: {macro_auc:.2f}\n' #for dissertation plot
            # f'Micro AUC: {micro_auc:.2f}\n' #for dissertation plot
            f'Average AUC: {average_auc:.2f}\n'
        )

        # Compute and add sample counts to the legend
        class_counts_train = y_train.value_counts()
        class_counts_test = y_test.value_counts()

        #for dissertation plot
        # for class_label in classes:
        #     train_count = class_counts_train.get(class_label, 0)
        #     test_count = class_counts_test.get(class_label, 0)
        #     total_count = train_count + test_count
        #     legend_text += f"'{class_label}': {train_count} training samples, {test_count} testing samples, {total_count} total\n"


        plt.plot([], [], ' ', label=legend_text)  # Add the text to the legend

        # Plot ROC curve for each class
        for i, class_label in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
            plt.plot(fpr, tpr, label=f'{class_label} (area = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title(f'Receiver Operating Characteristic (ROC) Curve for {target_name}')
    plt.legend(loc='lower right')
    plt.show()

    return average_auc

def plot_auc_heatmap(auc_dict, save_figure_path=None):
    """
    Generates a heatmap from a dictionary containing AUC values.

    Parameters:
    auc_dict (dict): A dictionary where keys are in the format '{Feature}_{Model}_auc' 
                     or '{Feature}_{Model}_LOOCV_auc' and values are AUC scores.
    """
    data = []

    for key, value in auc_dict.items():
        parts = key.split('_')
        
        # Case: If LOOCV is in the key, we need to adjust the extraction
        if "LOOCV" in parts:
            model = parts[-3]  # Model is the third last element
            feature = "_".join(parts[:-3])  # Feature is everything before model
        else:
            model = parts[-2]  # Model is the second last element
            feature = "_".join(parts[:-2])  # Feature is everything before model
        
        # Special case handling for cfRNA
        if key.startswith("cfRNA"):
            feature = "cfRNA"

        data.append((feature, model, value))

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['Feature', 'Model', 'AUC'])

    # Check for duplicates and fix them
    if df.duplicated(subset=['Feature', 'Model']).any():
        print("Warning: Duplicate feature-model pairs detected. Removing duplicates.")
        df = df.groupby(['Feature', 'Model'], as_index=False).mean()  # Take the mean AUC if duplicates exist

    # Pivot table for heatmap
    heatmap_data = df.pivot(index='Feature', columns='Model', values='AUC')

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="viridis", fmt=".2f")
    #plt.title("AUC Heatmap for Features and Models")
    plt.xlabel("")
    plt.ylabel("")
    if save_figure_path:
        plt.savefig(save_figure_path, dpi=300, bbox_inches='tight')
    plt.show()


