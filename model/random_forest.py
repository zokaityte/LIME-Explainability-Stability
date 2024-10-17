import csv
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
import numpy as np

from sklearn.tree import export_graphviz
import pydot
from io import StringIO
from PIL import Image

from common.generic import printc
from common.generic import pemji


class RandomForestClassifierModel:
    def __init__(self, *args, **kwargs):
        self.output_path = None
        self.training_params = kwargs
        self.model = RandomForestClassifier(*args, **kwargs)

    def generate_output_path(self, path):
        # Start the output path with the base path
        output_path = f'{path}/rf/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        #print(self.training_params)
        # Iterate over the kwargs and append them to the output path in the specified format
        for key, value in self.training_params.items():
            output_path += f'{key}_{value}'

        self.output_path = output_path

    def save_tree_image(self, train_labels, train_features, train_x, train_y, tree_index=0):
        """
        Saves an image of a specified decision tree from a trained Random Forest model.

        Parameters:
        - model: The trained Random Forest model.
        - train_x: The feature data used for training, to get feature names.
        - train_y: The target labels used for training, to get class names.
        - tree_index: The index of the tree to visualize (default is 0).
        - filename: The name of the file to save the tree image (default is 'tree.png').
        """
        
        # Extract the tree from the Random Forest
        tree = self.model.estimators_[tree_index]
        filename = f"{self.output_path}.png"
        
        # Export tree to Graphviz dot format
        dot_data = StringIO()
        export_graphviz(tree, out_file=dot_data, filled=True, rounded=True,
                        feature_names=train_features,  # Feature names for plotting
                        class_names=train_labels,  # Class names
                        special_characters=True)
        
        # Use pydot to convert the dot file to an image
        (graph,) = pydot.graph_from_dot_data(dot_data.getvalue())
        graph.write_png(filename)

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
        self.save_confusion_matrix_image(cm, f'{self.output_path}_confm.png')

        # Export metrics to CSV
        self.export_metrics_to_csv(f'{self.output_path}.csv', accuracy, precision, recall, f1, cm)
        
        printc(f"{pemji('rocket')} Trained RF metrics: Accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}", 'v')

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

    def export_metrics_to_csv(self, filename, accuracy, precision, recall, f1, cm):
        # Convert the training params to a string format
        params_str = '/'.join(f'{key}={value}' for key, value in self.training_params.items())
        delimiter = '/'
        # Create a list of metrics and confusion matrix values
        data = [
            ["Training params", "Accuracy", "Precision", "Recall", "F1", "Confusion matrix"],
            [params_str, accuracy, precision, recall, f1, delimiter.join(map(str, cm.tolist()))]
            # Use "-" if no weight or training param is available
        ]
        # To later readback conf matrix:
        # df_read = pd.read_csv('metrics_and_confusion_matrix.csv', sep=',', header=None)
        # parsed_cm = list(map(int, df_read['ConfusionMatrix'][0].split('/')))

        # Export to CSV
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(data)

    def save(self):
        joblib.dump(self.model, f'{self.output_path}.pkl')

    def load(self, path):
        self.model = joblib.load(path)