import csv
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
import numpy as np

from utils.print_utils import printc
from utils.print_utils import pemji


class LogisticRegressionModel:
    def __init__(self, *args, **kwargs):
        self.test_output_path = None
        self.val_output_path = None
        self.training_params = kwargs
        self.model = LogisticRegression(*args, **kwargs)

    def generate_output_path(self, path):
        # Start the output path with the base path
        output_path = f'{path}/logreg/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Append the kwargs to the output path in the specified format
        for key, value in self.training_params.items():
            output_path += f'{key}_{value}'

        self.test_output_path = f'{output_path}_test'
        self.val_output_path = f'{output_path}_val'

    def train(self, train_x, train_y, val_x=None, val_y=None):
        self.model.fit(train_x, train_y)

    def evaluate(self, test_x, test_y, current_timestamp, is_test=True):
        y_pred = self.model.predict(test_x)

        cm = confusion_matrix(test_y, y_pred)

        # weighted stats
        class_counts = np.sum(cm, axis=1)
        total_instances = np.sum(class_counts)
        class_percentages = class_counts / total_instances
        sample_weights = np.array([class_percentages[label] for label in test_y])
        weighted_accuracy = accuracy_score(test_y, y_pred, sample_weight=sample_weights)

        weighted_precision = precision_score(test_y, y_pred, average='weighted')
        weighted_recall = recall_score(test_y, y_pred, average='weighted')
        weighted_f1 = f1_score(test_y, y_pred, average='weighted')

        # macro stats
        accuracy = accuracy_score(test_y, y_pred)
        macro_precision = precision_score(test_y, y_pred, average='macro')
        macro_recall = recall_score(test_y, y_pred, average='macro')
        macro_f1 = f1_score(test_y, y_pred, average='macro')

        suffix = 'test' if is_test else 'val'
        path = self.test_output_path if is_test else self.val_output_path
        # Save confusion matrix as image
        self.save_confusion_matrix_image(cm, f'{path}.png')

        # Export metrics to CSV (including weighted accuracy)
        self.export_metrics_to_csv(f'{path}.csv', suffix,
                                   accuracy, weighted_accuracy,
                                   weighted_precision, weighted_recall, weighted_f1,
                                   macro_precision, macro_recall, macro_f1,
                                   cm, current_timestamp)

        # Print the metrics for both weighted and non-weighted
        printc(f"{pemji('rocket')} Trained Logistic Regression metrics:\n"
               f"Accuracy: {accuracy}, Weighted Accuracy: {weighted_accuracy}\n"
               f"Weighted -> Precision: {weighted_precision}, Recall: {weighted_recall}, F1: {weighted_f1}\n"
               f"Macro (Non-weighted) -> Precision: {macro_precision}, Recall: {macro_recall}, F1: {macro_f1}", 'v')

        # Return all the metrics for further use
        return accuracy, weighted_accuracy, weighted_precision, weighted_recall, weighted_f1, macro_precision, macro_recall, macro_f1, cm

    def save_confusion_matrix_image(self, cm, filename):
        # Normalize the confusion matrix (optional)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by row

        plt.figure(figsize=(12, 10))  # Increase the figure size for better spacing
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', annot_kws={"size": 8}, cbar=True)  # Add colorbar and increase font size
        plt.title('Confusion Matrix (Normalized)', fontsize=18)
        plt.xlabel('Predicted', fontsize=16)
        plt.ylabel('True', fontsize=16)

        # Rotate tick labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)

        # Adjust layout to avoid overlap
        plt.tight_layout()

        # Save the plot
        plt.savefig(filename)
        plt.close()

    def export_metrics_to_csv(self, filename, suffix, accuracy, weighted_accuracy,
                              weighted_precision, weighted_recall, weighted_f1,
                              macro_precision, macro_recall, macro_f1,
                              cm, current_timestamp):
        # Convert the training params to a string format
        params_str = '/'.join(f'{key}={value}' for key, value in self.training_params.items())
        delimiter = '/'
        # Create a list of metrics and confusion matrix values
        data = [
            ["test/val", "Timestamp", "Training params", "Accuracy weighted", "Precision weighted", "Recall weighted", "F1 weighted", "Accuracy", "Precision", "Recall", "F1", "Confusion matrix"],
            [suffix, current_timestamp, params_str, weighted_accuracy, weighted_precision, weighted_recall, weighted_f1, accuracy, macro_precision, macro_recall, macro_f1, delimiter.join(map(str, cm.tolist()))]
        ]

        # Export to CSV
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(data)

    def save(self):
        joblib.dump(self.model, f'{self.val_output_path}.pkl')

    def load(self, path):
        self.model = joblib.load(path)