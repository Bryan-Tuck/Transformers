import os
import re
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00",
                        "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))


def get_method_name_from_file(file: str) -> str:
    """
    :param file: the file name
    :returns: the method name
    """
    match = re.search(
        r'olid_(\d+)', file)  # get the method number from the file name
    if match:  # check if the method number is found
        return match.group(1)
    else:
        # raise an error if the method number is not found
        raise ValueError(f'Invalid file name format: {file}')


def get_classification_report(y_test, y_pred):
    """
    :param y_test: the true labels
    :param y_pred: the predicted labels
    :param return: a dictionary containing the classification report
    """
    # Get the classification report
    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=1)
    print(f"Classification report:\n{report}")
    metrics = {}
    metrics["precision_macro"] = report["macro avg"]["precision"]
    metrics["recall_macro"] = report["macro avg"]["recall"]
    metrics["f1_macro"] = report["macro avg"]["f1-score"]
    metrics["precision_weighted"] = report["weighted avg"]["precision"]
    metrics["recall_weighted"] = report["weighted avg"]["recall"]
    metrics["f1_weighted"] = report["weighted avg"]["f1-score"]
    metrics["accuracy_score"] = accuracy_score(y_test, y_pred)
    metrics["matthews_correlation_coefficient"] = matthews_corrcoef(y_test, y_pred)
    return metrics


def save_classification_report(method, metrics, save_dir, append=False):
    """
    :param method: the method name
    :param metrics: a dictionary containing the classification report
    :param save_dir: the directory where the results will be saved
    :param append: whether to append to the output file or overwrite it (default: False)
    """
    # File I/O error handling
    try:
        # Create a DataFrame to store the classification report
        df = pd.DataFrame({
            "Method": [method],
            "Precision Macro": [metrics["precision_macro"]],
            "Recall Macro": [metrics["recall_macro"]],
            "F1 Macro": [metrics["f1_macro"]],
            "Precision Weighted": [metrics["precision_weighted"]],
            "Recall Weighted": [metrics["recall_weighted"]],
            "F1 Weighted": [metrics["f1_weighted"]],
            "Accuracy": [metrics["accuracy_score"]],
            "MCC": [metrics["matthews_correlation_coefficient"]]
        })

        # Check if the directory to save the report exists, create it if it does not
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Write the DataFrame to a CSV file
        file_path = os.path.join(save_dir, "report.csv")
        mode = "a" if append else "w"
        df.to_csv(file_path, mode=mode, header=not append, index=False)
        print(f"Classification report saved to {file_path}")
    except Exception as e:
        print(
            f"An error occurred while writing the classification report to {save_dir}: {e}")



def create_plots(method, train_loss_values, val_loss_values, train_acc_values, val_acc_values):
    """
    :param method: the method name
    :param train_loss_values: the training loss values
    :param val_loss_values: the validation loss values
    :param train_acc_values: the training accuracy values
    :param val_acc_values: the validation accuracy values
    """
    # Create the plots
    plt.style.use('seaborn')
    _, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Loss subplot
    axs[0].plot(train_loss_values, label='training loss')
    axs[0].plot(val_loss_values, label='validation loss')
    axs[0].set_title("Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Accuracy subplot
    axs[1].plot(train_acc_values, label='training accuracy')
    axs[1].plot(val_acc_values, label='validation accuracy')
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    plt.suptitle(f'Training and Validation Loss/Accuracy for {method}')

    # Save the plots to a folder with the method name
    plot_directory = f'E:/Transformers/plots'
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    plt.savefig(f'{plot_directory}\\{method}_plots.png')
