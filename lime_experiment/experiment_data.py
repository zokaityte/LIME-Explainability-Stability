import os

import pandas as pd


class ExperimentData:

    def __init__(self, dataset_path: str, label_names: list, categorical_columns_names=None):
        self.dataset_path = dataset_path
        self.train_data_csv_path = os.path.join(self.dataset_path, "train_data.csv")
        self.val_data_csv_path = os.path.join(self.dataset_path, "val_data.csv")
        self.test_data_csv_path = os.path.join(self.dataset_path, "test_data.csv")
        self.label_names = label_names
        self.categorical_columns_names = categorical_columns_names if categorical_columns_names else []

        self.training_data = pd.read_csv(self.train_data_csv_path)
        self.features_np = self.training_data[self.training_data.columns[:-1]].to_numpy()
        self.labels_np = self.training_data[self.training_data.columns[-1]]

    def get_training_data(self):
        print("Warning: training data not implemented")

        return self.features_np

    def get_training_labels(self):
        print("Warning: training labels not implemented")
        return self.labels_np

    def get_random_test_instance(self, random_seed):
        print("Warning: random test instance not implemented")
        return self.features_np[0, :]

    def get_num_classes(self):
        return len(set(self.labels_np))

    def get_class_names(self):
        return self.label_names

    def get_categorical_features(self):
        """Returns the indexes of categorical columns based on the column names."""

        df = pd.read_csv(self.train_data_csv_path, nrows=0)

        indexes = []
        for col_name in self.categorical_columns_names:
            if col_name in df.columns:
                indexes.append(df.columns.get_loc(col_name))

        return indexes

    def get_categorical_names(self):
        print("Warning: categorical names not implemented")
        return None

    def get_feature_names(self):
        df_header = pd.read_csv(self.train_data_csv_path, nrows=0)
        return df_header.columns[:-1].tolist()

    def get_count_of_categorical_features(self):
        return len(self.categorical_columns_names)
