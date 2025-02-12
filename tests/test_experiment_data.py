import os
import unittest

import numpy as np
import pandas as pd

from lime_experiment.experiment_data import ExperimentData


class TestExperimentData(unittest.TestCase):

    def setUp(self):
        self.dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./resources/sample_dataset_1"))
        self.dataset_with_nan_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./resources/sample_dataset_2"))
        self.common_parameters = {
            "label_names": ["Benign", "FTP-BruteForce", "SSH-Bruteforce"],
            "categorical_columns_names": ["Fwd PSH Flags", "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt",
                                          "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt", "ECE Flag Cnt"]
        }


    def test_get_random_test_instance(self):
        experiment_data = ExperimentData(self.dataset_path, **self.common_parameters)
        experiment_data.get_random_test_instance(random_seed=42)

        self.assertEqual(experiment_data.random_test_row_index, 28)

    def test_get_random_test_instance_class(self):
        experiment_data = ExperimentData(self.dataset_path, **self.common_parameters)


        instance = experiment_data.get_random_test_instance(random_seed=42, class_label=1)
        expected_instance = pd.read_csv(experiment_data._test_data_csv_path, engine="pyarrow").iloc[
                                experiment_data.random_test_row_index].to_numpy()[:-1]

        self.assertTrue(np.array_equal(instance, expected_instance))
        self.assertEqual(experiment_data.random_test_row_index, 115)
        self.assertEqual(experiment_data.random_test_row_label, 1)

    def test_get_random_test_instance_impute(self):
        experiment_data = ExperimentData(self.dataset_with_nan_path, **self.common_parameters)

        instance = experiment_data.get_random_test_instance(random_seed=42, class_label=2)
        imputed_value = instance[49]
        self.assertEqual(97.4458913120979, imputed_value)

        self.assertEqual(experiment_data.random_test_row_index, 141)
        self.assertEqual(experiment_data.random_test_row_label, 2)






