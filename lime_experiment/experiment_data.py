import os
import random
import numpy as np
from sklearn.impute import SimpleImputer

import pandas as pd

CHUNK_SIZE = 10000


class ExperimentData:

    def __init__(self, dataset_path: str, label_names: list, categorical_columns_names=None):
        self._dataset_path = dataset_path
        self._train_data_csv_path = os.path.join(self._dataset_path, "train.csv")
        self._val_data_csv_path = os.path.join(self._dataset_path, "val.csv")
        self._test_data_csv_path = os.path.join(self._dataset_path, "test.csv")
        self._label_names = label_names
        self._categorical_columns_names = categorical_columns_names if categorical_columns_names else []
        self._column_names = self._get_column_names()
        self.random_text_row_index = None

    def get_training_data(self):
        # TODO Check if validation split is needed

        # float64 limit into nan
        train_data = pd.read_csv(self._train_data_csv_path, engine="pyarrow", usecols=self._column_names[:-1])
        train_data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Nans as mean
        imputer = SimpleImputer(strategy='mean')
        train_data = imputer.fit_transform(train_data)

        return train_data

    def get_training_labels(self):
        return pd.read_csv(self._train_data_csv_path, engine="pyarrow", usecols=[self._column_names[-1]]).to_numpy()

    def get_random_test_instance(self, random_seed):
        # Set the random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)

        # Get the total number of rows (excluding the header)
        with open(self._test_data_csv_path, 'r') as f:
            row_count = sum(1 for _ in f) - 1  # Subtract 1 for the header row

        # Select a random row index (between 1 and row_count, skipping the header)
        random_row_index = random.randint(1, row_count)
        self.random_text_row_index = random_row_index

        # Read only the selected row (skip all rows except the randomly selected one)
        random_row = pd.read_csv(self._test_data_csv_path, skiprows=range(1, random_row_index), nrows=1)
        random_row_features = random_row.iloc[:, :-1]
        random_row_features_numpy = random_row_features.to_numpy().flatten()
        return random_row_features_numpy

    def get_num_classes(self):
        return len(self._label_names)

    def get_class_names(self):
        return self._label_names

    def get_categorical_features(self):
        """Returns the indexes of categorical columns based on the column names."""

        df = pd.read_csv(self._train_data_csv_path, nrows=0)

        indexes = []
        for col_name in self._categorical_columns_names:
            if col_name in df.columns:
                indexes.append(df.columns.get_loc(col_name))

        return indexes

    def get_categorical_names(self):
        """Returns a dictionary with the index of categorical columns and their unique values from all files."""
        files = [self._train_data_csv_path, self._val_data_csv_path, self._test_data_csv_path]
        df_header = pd.read_csv(self._train_data_csv_path, nrows=0)
        categorical_dict = {}

        for file_path in files:
            chunk_size = CHUNK_SIZE
            for chunk in pd.read_csv(file_path, usecols=self._categorical_columns_names, chunksize=chunk_size):
                for col_name in self._categorical_columns_names:
                    col_index = df_header.columns.get_loc(col_name)
                    if col_index not in categorical_dict:
                        categorical_dict[col_index] = set(chunk[col_name].dropna().unique())
                    else:
                        categorical_dict[col_index].update(chunk[col_name].dropna().unique())

        categorical_dict = {index: list(values) for index, values in categorical_dict.items()}
        return categorical_dict

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

    def get_feature_count(self):
        return len(self.get_feature_names())

    def get_dataset_path(self):
        return self._dataset_path
