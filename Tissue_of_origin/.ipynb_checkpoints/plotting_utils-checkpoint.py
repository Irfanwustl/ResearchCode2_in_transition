import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize



def plot_roc_curve(y_test, y_pred_proba, y_train, target_name, classes):
    plt.figure(figsize=(12, 10))
    
    if len(y_train.unique()) == 2:
        # Binary classification case
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        plt.plot(fpr, tpr, label=f'{target_name} ROC curve (AUC = {roc_auc:.2f})')
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
        
        average_auc /= len(classes)  # Calculate the average AUC

        # Initialize the legend text
        legend_text = (
            f'Macro AUC: {macro_auc:.2f}\n'
            f'Micro AUC: {micro_auc:.2f}\n'
            f'Average AUC: {average_auc:.2f}\n'
        )

        # Compute and add sample counts to the legend
        class_counts_train = y_train.value_counts()
        class_counts_test = y_test.value_counts()

        for class_label in classes:
            train_count = class_counts_train.get(class_label, 0)
            test_count = class_counts_test.get(class_label, 0)
            total_count = train_count + test_count
            legend_text += f"'{class_label}': {train_count} training samples, {test_count} testing samples, {total_count} total\n"

        plt.plot([], [], ' ', label=legend_text)  # Add the text to the legend

        for i, class_label in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
            plt.plot(fpr, tpr, label=f'ROC curve of class {class_label} (area = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {target_name}')
    plt.legend(loc='lower right')
    plt.show()

def plot_precision_recall_curve(y_test, y_pred_proba, y_train, target_name, classes):
    plt.figure(figsize=(12, 10))
    
    if len(y_train.unique()) == 2:
        # Binary classification case
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
        pr_auc = average_precision_score(y_test, y_pred_proba[:, 1])
        plt.plot(recall, precision, label=f'{target_name} PR curve (AUC = {pr_auc:.2f})')
    else:
        # Multiclass case
        y_test_bin = label_binarize(y_test, classes=classes)
        average_pr_auc = 0
        for i, class_label in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
            pr_auc = average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])
            average_pr_auc += pr_auc
            plt.plot(recall, precision, label=f'{target_name} PR curve for {class_label} (AUC = {pr_auc:.2f})')
        average_pr_auc /= y_test_bin.shape[1]
        plt.plot([], [], ' ', label=f'Average PR AUC = {average_pr_auc:.2f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {target_name}')
    plt.legend(loc='lower left')
    plt.show()

def plot_confusion_matrix(y_test, y_pred, target_name, classes):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Confusion Matrix for {target_name}')
    plt.show()

def plot_classification_results(model, X_test, y_test, y_train, target_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Get class labels from the model
    classes = model.classes_
    
    # Plot ROC curve
    plot_roc_curve(y_test, y_pred_proba, y_train, target_name, classes)
    
    # Plot Precision-Recall curve
    #plot_precision_recall_curve(y_test, y_pred_proba, y_train, target_name, classes)
    
    # Plot Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, target_name, classes)
