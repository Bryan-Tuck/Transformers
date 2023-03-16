from sklearn.metrics import classification_report, accuracy_score, matthews_corrcoef
from utils import get_classification_report, save_classification_report
import sys
import os
import pandas as pd
import numpy as np

# Add the path to the utils.py file to the system path
sys.path.append('E:\Transformers')

def test_get_classification_report():
    # define sample input data
    y_test = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]

    # get the classification report
    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=1)

    # get the expected output
    expected_output = {}
    expected_output["precision_macro"] = "{:.4f}".format(
        report["macro avg"]["precision"])
    expected_output["recall_macro"] = "{:.4f}".format(
        report["macro avg"]["recall"])
    expected_output["f1_macro"] = "{:.4f}".format(
        report["macro avg"]["f1-score"])
    expected_output["precision_weighted"] = "{:.4f}".format(
        report["weighted avg"]["precision"])
    expected_output["recall_weighted"] = "{:.4f}".format(
        report["weighted avg"]["recall"])
    expected_output["f1_weighted"] = "{:.4f}".format(
        report["weighted avg"]["f1-score"])
    expected_output["accuracy_score"] = "{:.4f}".format(
        accuracy_score(y_test, y_pred))
    expected_output["matthews_correlation_coefficient"] = "{:.4f}".format(
        matthews_corrcoef(y_test, y_pred))

    # get the actual output
    actual_output = get_classification_report(y_test, y_pred)

    # check that the actual output matches the expected output
    assert actual_output == expected_output


def test_save_classification_report(tmp_path):
    # define sample input data
    method = "sample method"
    metrics = {
        "precision_macro": "0.8750",
        "recall_macro": "0.8333",
        "f1_macro": "0.8438",
        "precision_weighted": "0.8750",
        "recall_weighted": "0.8333",
        "f1_weighted": "0.8438",
        "accuracy_score": "0.8333",
        "matthews_correlation_coefficient": "0.6250"
    }
    save_path = os.path.join(tmp_path, "test_report.csv")

    # call the function to save the classification report
    save_classification_report(method, metrics, save_path, append=False)

    # check that the output file was created and contains the expected data
    expected_output = pd.DataFrame({
        "Method": [method],
        "Precision Macro": [float(metrics["precision_macro"])],
        "Recall Macro": [float(metrics["recall_macro"])],
        "F1 Macro": [float(metrics["f1_macro"])],
        "Precision Weighted": [float(metrics["precision_weighted"])],
        "Recall Weighted": [float(metrics["recall_weighted"])],
        "F1 Weighted": [float(metrics["f1_weighted"])],
        "Accuracy": [float(metrics["accuracy_score"])],
        "MCC": [float(metrics["matthews_correlation_coefficient"])]
    })
    assert os.path.isfile(save_path)
    actual_output = pd.read_csv(save_path)
    pd.testing.assert_frame_equal(actual_output, expected_output)
