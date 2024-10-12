import pandas as pd


class ExperimentData:

    def __init__(self, train_data_csv_path, val_data_csv_path, test_data_csv_path):
        self.train_data_csv_path = train_data_csv_path
        self.val_data_csv_path = val_data_csv_path
        self.test_data_csv_path = test_data_csv_path

        self.training_data = pd.read_csv(self.train_data_csv_path)
        self.features_np = self.training_data[self.training_data.columns[:-1]].to_numpy()
        self.labels_np = self.training_data[self.training_data.columns[-1]]

    def get_training_data(self):
        print("Warning: training data not implemented")

        return self.features_np

    def get_training_labels(self):
        print("Warning: training labels not implemented")
        return self.labels_np

    def get_random_test_instance(self):
        print("Warning: random test instance not implemented")
        return self.features_np[0, :]

    def get_num_classes(self):
        return len(set(self.labels_np))

    def get_class_names(self):
        print("Warning: class names not implemented")
        return None

    def get_categorical_features(self):
        print("Warning: categorical features not implemented")
        return None

    def get_categorical_names(self):
        print("Warning: categorical names not implemented")
        return None

    def get_feature_names(self):
        print("Warning: feature names not implemented")
        return None
