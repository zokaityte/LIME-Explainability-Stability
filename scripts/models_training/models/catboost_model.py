import csv
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


class CatBoostClassifierModel:
    def __init__(self, *args, **kwargs):
        self.val_output_path = None
        self.test_output_path = None
        self.output_path = None
        self.training_params = kwargs
        self.model = CatBoostClassifier(*args, verbose=0, **kwargs, allow_writing_files=False)

    def generate_output_path(self, path):
        output_path = f'{path}/catboost/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for key, value in self.training_params.items():
            output_path += f'{key}_{value}'

        self.test_output_path = f'{output_path}_test'
        self.val_output_path = f'{output_path}_val'

    def train(self, train_x, train_y, val_x=None, val_y=None, early_stopping_rounds=None):
        eval_set = None
        if val_x is not None and val_y is not None:
            eval_set = [(val_x, val_y)]

        self.model.fit(
            train_x,
            train_y,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=True
        )

    def evaluate(self, test_x, test_y, current_timestamp, is_test=True):
        y_pred = self.model.predict(test_x)

        cm = confusion_matrix(test_y, y_pred)
        class_counts = np.sum(cm, axis=1)
        total_instances = np.sum(class_counts)
        class_percentages = class_counts / total_instances
        sample_weights = np.array([class_percentages[label] for label in test_y])
        weighted_accuracy = accuracy_score(test_y, y_pred, sample_weight=sample_weights)
        weighted_precision = precision_score(test_y, y_pred, average='weighted')
        weighted_recall = recall_score(test_y, y_pred, average='weighted')
        weighted_f1 = f1_score(test_y, y_pred, average='weighted')
        accuracy = accuracy_score(test_y, y_pred)
        macro_precision = precision_score(test_y, y_pred, average='macro')
        macro_recall = recall_score(test_y, y_pred, average='macro')
        macro_f1 = f1_score(test_y, y_pred, average='macro')

        suffix = 'test' if is_test else 'val'
        path = self.test_output_path if is_test else self.val_output_path

        self.save_confusion_matrix_image(cm, f'{path}.png')
        self.export_metrics_to_csv(
            f'{path}.csv',
            suffix,
            accuracy,
            weighted_accuracy,
            weighted_precision,
            weighted_recall,
            weighted_f1,
            macro_precision,
            macro_recall,
            macro_f1,
            cm,
            current_timestamp
        )

    def save_confusion_matrix_image(self, cm, filename):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            annot_kws={"size": 8},
            cbar=True
        )
        plt.title('Confusion Matrix (Normalized)', fontsize=18)
        plt.xlabel('Predicted', fontsize=16)
        plt.ylabel('True', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def export_metrics_to_csv(
        self,
        filename,
        suffix,
        accuracy,
        weighted_accuracy,
        weighted_precision,
        weighted_recall,
        weighted_f1,
        macro_precision,
        macro_recall,
        macro_f1,
        cm,
        current_timestamp
    ):
        params_str = '/'.join(f'{key}={value}' for key, value in self.training_params.items())
        delimiter = '/'
        data = [
            [
                "test/val",
                "Timestamp",
                "Training params",
                "Accuracy weighted",
                "Precision weighted",
                "Recall weighted",
                "F1 weighted",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "Confusion matrix"
            ],
            [
                suffix,
                current_timestamp,
                params_str,
                weighted_accuracy,
                weighted_precision,
                weighted_recall,
                weighted_f1,
                accuracy,
                macro_precision,
                macro_recall,
                macro_f1,
                delimiter.join(map(str, cm.tolist()))
            ]
        ]

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(data)

    def save(self):
        joblib.dump(self.model, f'{self.val_output_path}.pkl')

    def load(self, path):
        self.model = joblib.load(path)
