import csv
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os

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

        print(self.training_params)
        # Iterate over the kwargs and append them to the output path in the specified format
        for key, value in self.training_params.items():
            output_path += f'{key}_{value}'

        self.output_path = output_path

    def train(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def evaluate(self, test_x, test_y):
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
        self.export_metrics_to_csv(f'{self.output_path}.csv', accuracy, precision, recall, f1, cm)

        # Return all the metrics for further use
        return accuracy, precision, recall, f1, cm

    def save_confusion_matrix_image(self, cm, filename):
        # Plot confusion matrix using seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(filename)
        plt.close()

    def export_metrics_to_csv(self, filename, accuracy, precision, recall, f1, cm):
        # Convert the training params to a string format
        params_str = ', '.join(f'{key}={value}' for key, value in self.training_params.items())
        # Create a list of metrics and confusion matrix values
        data = [
            ["Training params", "Accuracy", "Precision", "Recall", "F1", "Confusion matrix"],
            [params_str, accuracy, precision, recall, f1, cm.tolist()]
        ]

        # Export to CSV
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(data)

    def save(self):
        joblib.dump(self.model, f'{self.output_path}.pkl')

    def load(self, path):
        self.model = joblib.load(path)