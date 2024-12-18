import os
import random

import pandas as pd


class ExperimentData:

    def __init__(self, dataset_path: str, label_names: list, categorical_columns_names=None):
        self._dataset_path = dataset_path
        self._train_data_csv_path = os.path.join(self._dataset_path, "train.csv")
        self._val_data_csv_path = os.path.join(self._dataset_path, "val.csv")
        self._test_data_csv_path = os.path.join(self._dataset_path, "test.csv")
        self._label_names = label_names
        self._categorical_columns_names = categorical_columns_names if categorical_columns_names else []
        self._column_names = self._get_column_names()
        self.random_test_row_index = None
        self.random_test_row_label = None

        self._categorical_features = None
        self._categorical_names = None

    def get_training_data(self):
        train_data = pd.read_csv(self._train_data_csv_path, engine="pyarrow", usecols=self._column_names[:-1])
        return train_data.to_numpy()

    def get_training_labels(self):
        return pd.read_csv(self._train_data_csv_path, engine="pyarrow", usecols=[self._column_names[-1]]).to_numpy()

    def get_random_test_instance(self, random_seed, class_label):
        # Set the random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)

        test_data = pd.read_csv(self._test_data_csv_path, engine="pyarrow")

        # Filter by class label
        if class_label is not None:

            # Check if the class label exists in the dataset
            if class_label not in test_data[:, -1]:
                raise ValueError(f"The class label {class_label} does not exist in the dataset.")

            test_data = test_data[test_data[:, -1] == class_label]

        # Select a random row index
        random_row_index = random.randint(0, test_data.shape[0])
        self.random_test_row_index = random_row_index
        self.random_test_row_label = test_data.iloc[random_row_index, -1]

        # Select the features of the random row
        random_row_features = test_data.iloc[random_row_index, :-1]
        random_row_features_numpy = random_row_features.to_numpy().flatten()
        return random_row_features_numpy

    def get_num_classes(self):
        return len(self._label_names)

    def get_class_names(self):
        return self._label_names

    def get_categorical_features(self):
        """Returns the indexes of categorical columns based on the column names."""
        if self._categorical_features:
            return self._categorical_features

        df = pd.read_csv(self._train_data_csv_path, nrows=0)

        indexes = []
        for col_name in self._categorical_columns_names:
            if col_name in df.columns:
                indexes.append(df.columns.get_loc(col_name))

        self._categorical_features = indexes
        return self._categorical_features

    def _get_column_names(self):
        df_header_1 = pd.read_csv(self._train_data_csv_path, nrows=0)
        df_header_2 = pd.read_csv(self._val_data_csv_path, nrows=0)
        df_header_3 = pd.read_csv(self._test_data_csv_path, nrows=0)
        if df_header_1.equals(df_header_2) and df_header_2.equals(df_header_3):
            return df_header_1.columns.tolist()
        else:
            raise ValueError("Feature names are not consistent across the datasets.")

    def get_feature_names(self):
        return self._get_column_names()[:-1]

    def get_categorical_features_count(self):
        return len(self._categorical_columns_names)

    def get_categorical_features_names(self):
        return self._categorical_columns_names

    def get_feature_count(self):
        return len(self.get_feature_names())

    def get_dataset_name(self):
        return os.path.basename(self._dataset_path)
