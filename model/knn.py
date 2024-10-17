import csv
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
import numpy as np

from common.generic import printc
from common.generic import pemji


class KNeighborsClassifierModel:
    def __init__(self, *args, **kwargs):
        self.output_path = None
        self.training_params = kwargs
        self.model = KNeighborsClassifier(*args, **kwargs)

    def generate_output_path(self, path):
        # Start the output path with the base path
        output_path = f'{path}/knn/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        #print(self.training_params)
        # Iterate over the kwargs and append them to the output path in the specified format
        for key, value in self.training_params.items():
            output_path += f'{key}_{value}'

        self.output_path = output_path

    def train(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def evaluate(self, test_x, test_y, current_timestamp):
        # Get the predictions
        y_pred = self.model.predict(test_x)

        # Calculate metrics
        accuracy = accuracy_score(test_y, y_pred)
        precision = precision_score(test_y, y_pred, average='weighted')
        recall = recall_score(test_y, y_pred, average='weighted')
        f1 = f1_score(test_y, y_pred, average='weighted')

        # Calculate confusion matrix
        cm = confusion_matrix(test_y, y_pred)

        # Save confusion matrix as image
        self.save_confusion_matrix_image(cm, f'{self.output_path}.png')

        # Export metrics to CSV
        self.export_metrics_to_csv(f'{self.output_path}.csv', accuracy, precision, recall, f1, cm, current_timestamp)
        
        printc(f"{pemji('rocket')} Trained KNN metrics: Accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}", 'v')

        # Return all the metrics for further use
        return accuracy, precision, recall, f1, cm

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

    def export_metrics_to_csv(self, filename, accuracy, precision, recall, f1, cm, current_timestamp):
        # Convert the training params to a string format
        params_str = '/'.join(f'{key}={value}' for key, value in self.training_params.items())
        delimiter = '/'
        # Create a list of metrics and confusion matrix values
        data = [
            ["Timestamp", "Training params", "Accuracy", "Precision weighted", "Recall weighted", "F1 weighted", "Confusion matrix"],
            [current_timestamp, params_str, accuracy, precision, recall, f1, delimiter.join(map(str, cm.tolist()))]
        ]

        # Export to CSV
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(data)

    def save(self):
        joblib.dump(self.model, f'{self.output_path}.pkl')

    def load(self, path):
        self.model = joblib.load(path)