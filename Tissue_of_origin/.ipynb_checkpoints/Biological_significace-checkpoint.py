import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

def plot_feature_boxplot_with_anova(df, feature_name, target_name='target', class_order=None, save_path=None):
    """
    This function generates a box plot for a given feature, separated by target classes.
    If a save_path is provided, the plot will be saved at the specified path.
    If no save_path is provided, the plot will be displayed.
    It also returns the p-value and F-statistic from an ANOVA test.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the feature and target.
    feature_name (str): The name of the feature to plot.
    target_name (str): The name of the target column, default is 'target'.
    class_order (list): An optional list specifying the order of the classes.
    save_path (str): The path to save the plot. If None, the plot will be displayed.
    
    Returns:
    float, float: F-statistic and p-value from the ANOVA test.
    """
    # Perform ANOVA test
    groups = [df[feature_name][df[target_name] == class_value] for class_value in df[target_name].unique()]
    anova_result = f_oneway(*groups)
    f_stat = anova_result.statistic
    p_value = anova_result.pvalue

    # Create the boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[target_name], y=df[feature_name], order=class_order)
    
    # Add the p-value from ANOVA in the title
    plt.title(f'Boxplot of {feature_name} by {target_name} (ANOVA p-value: {p_value:.4f})')
    plt.xlabel(target_name)
    plt.ylabel(feature_name)
    
    if save_path:
        # Save the plot and close it
        plt.savefig(save_path)
        plt.close()
    else:
        # Show the plot if no save_path is provided
        plt.show()
    
    # Return F-statistic and p-value
    return f_stat, p_value
