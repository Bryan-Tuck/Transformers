import os
import re
import csv
from sklearn.metrics import classification_report
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
    :return: the method name
    """
    match = re.search(
        r'method_(\d+)', file)  # get the method number from the file name
    if match:  # check if the method number is found
        return match.group(1)
    else:
        # raise an error if the method number is not found
        raise ValueError(f'Invalid file name format: {file}')


def get_classification_report(y_test, y_pred):
    """
    params: y_test: the true labels
            y_pred: the predicted labels
            return: precision_macro, recall_macro, f1_macro, precision_weighted, recall_weighted, f1_weighted, accuracy_score
    """
    # Get the classification report
    report = classification_report(y_test, y_pred, output_dict=True, digits=4)
    precision_macro = report["macro avg"]["precision"]
    recall_macro = report["macro avg"]["recall"]
    f1_macro = report["macro avg"]["f1-score"]
    precision_weighted = report["weighted avg"]["precision"]
    recall_weighted = report["weighted avg"]["recall"]
    f1_weighted = report["weighted avg"]["f1-score"]
    accuracy_score = report["accuracy"]
    matthews_correlation_coefficient = matthews_corrcoef(y_test, y_pred)
    return precision_macro, recall_macro, f1_macro, precision_weighted, recall_weighted, f1_weighted, accuracy_score, matthews_correlation_coefficient


def save_classification_report(method, precision_macro, recall_macro, f1_macro, precision_weighted, recall_weighted, f1_weighted, accuracy_score, matthews_correlation_coefficient):
    """
    :param method: the method name
    :param precision_macro: the precision macro
    :param recall_macro: the recall macro
    :param f1_macro: the f1 macro
    :param precision_weighted: the precision weighted
    :param recall_weighted: the recall weighted
    :param f1_weighted: the f1 weighted
    :param accuracy_score: the accuracy score
    """
    with open("results\\classification_report.csv", "a", newline="") as f:
        # Create a CSV writer
        writer = csv.DictWriter(f, fieldnames=['Method', 'Precision Macro',
                                               'Recall Macro', 'F1 Macro', 'Precision Weighted',
                                               'Recall Weighted', 'F1 Weighted', 'Accuracy', 'MCC'])

        # Write the header row if the file is empty
        if f.tell() == 0:
            writer.writeheader()

        # Write the data row
        writer.writerow({'Method': method, 'Precision Macro': precision_macro,
                         'Recall Macro': recall_macro, 'F1 Macro': f1_macro,
                         'Precision Weighted': precision_weighted, 'Recall Weighted': recall_weighted,
                         'F1 Weighted': f1_weighted, 'Accuracy': accuracy_score, 'MCC': matthews_correlation_coefficient})


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
    plot_directory = f'plots'
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    plt.savefig(f'{plot_directory}\\{method}_plots.png')
