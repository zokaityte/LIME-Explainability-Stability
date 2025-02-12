import os
import random

import pandas as pd
from sklearn.impute import SimpleImputer


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

    def get_training_data(self, imputed=False):
        train_data = pd.read_csv(self._train_data_csv_path, engine="pyarrow", usecols=self._column_names[:-1])

        if imputed:
            train_data.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
            imputer = SimpleImputer(strategy="mean")
            train_data = imputer.fit_transform(train_data)
            return train_data

        return train_data.to_numpy()

    def get_test_data(self, imputed=False):
        test_data = pd.read_csv(self._test_data_csv_path, engine="pyarrow", usecols=self._column_names[:-1])

        if imputed:
            test_data.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
            imputer = SimpleImputer(strategy="mean")
            test_data = imputer.fit_transform(test_data)
            return test_data

        return test_data.to_numpy()

    def get_test_labels(self):
        return pd.read_csv(self._test_data_csv_path, engine="pyarrow", usecols=[self._column_names[-1]]).to_numpy()

    def get_training_labels(self):
        return pd.read_csv(self._train_data_csv_path, engine="pyarrow", usecols=[self._column_names[-1]]).to_numpy()

    def get_random_test_instance(self, random_seed, class_label=None):
        # Set the random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)

        test_x = self.get_test_data(imputed=True)
        test_y = self.get_test_labels().flatten()

        test_df = pd.DataFrame(test_x)
        test_df["original_index"] = test_df.index
        test_df["label"] = test_y

        # Filter by class label
        if class_label is not None:

            if class_label not in test_df["label"].unique():
                raise ValueError(f"The class label {class_label} does not exist in the test dataset.")

            test_df = test_df[test_df["label"] == class_label]

        # Select a random row index
        random_row = random.randint(0, test_df.shape[0])
        self.random_test_row_index = int(test_df.iloc[random_row]["original_index"])
        self.random_test_row_label = test_df.iloc[random_row]["label"]
        random_test_row_features = test_df.iloc[random_row].drop(["original_index", "label"]).to_numpy()

        # Select the features of the random row
        return random_test_row_features

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
