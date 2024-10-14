import os
import unittest

from lime_experiment.experiment import LimeExperiment
from lime_experiment.experiment_data import ExperimentData
from lime_experiment.explained_model import ExplainedModel
from lime_experiment.models import LimeExplainerConfig, LimeExperimentConfig


class TestLimeExperiment(unittest.TestCase):

    def setUp(self):
        """Setup common data for the test."""

        # Convert relative paths to absolute paths
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model_checkpoints/sample_dataset_1_rf_model.pkl"))
        dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/sample_dataset_1"))

        self.random_seed = 42  # Or None
        self.explained_model = ExplainedModel(model_path)
        self.experiment_data = ExperimentData(
            dataset_path,
            label_names=["Benign", "FTP-BruteForce", "SSH-Bruteforce"],
            categorical_columns_names=["Fwd PSH Flags", "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
                                       "ACK Flag Cnt", "URG Flag Cnt", "ECE Flag Cnt"],
        )

        self.explainer_config = LimeExplainerConfig(**{
            'kernel_width': None,
            'kernel': None,
            'sample_around_instance': False,
            'num_features': 5,
            'num_samples': 5000,
            'sampling_func': 'gaussian',
            'sampling_func_params': {}
        })

        self.times_to_run = 5

    def test_experiment_run(self):
        """Test the experiment run and verify explanations."""
        # Run experiment
        experiment = LimeExperiment(
            LimeExperimentConfig(
                self.explained_model, self.experiment_data, self.explainer_config,
                self.times_to_run, self.random_seed, save_explanations=False, save_results=False
            ),
        )
        experiment.run()

        # Expected explanation for the first class
        expected_explanation = [('ACK Flag Cnt=0', 0.06637493923858132),
                                ('Flow Pkts/s', 0.03886579142502093),
                                ('Init Fwd Win Byts', 0.023605242420647873),
                                ('Subflow Fwd Pkts', -0.023575766513791962),
                                ('Bwd Pkts/s', 0.02324818707547263)]

        # Verify explanations
        for explanation in experiment._explanations:
            actual_explanation = explanation.as_list(1)
            for i, (feature, value) in enumerate(expected_explanation):
                with self.subTest(i=i):
                    self.assertEqual(actual_explanation[i][0], feature)
                    self.assertAlmostEqual(actual_explanation[i][1], value, places=7)

        # Verify experiment results
        results = experiment.get_results()
        expected_results = {
            'number_of_features': 65,
            'number_of_categorical_features': 8,
            'explained_model_type': 'RandomForestClassifier',
            'times_explained': 5,
            'random_seed': 42,
            'kernel_width': 'default (sqrt(num_features) * 0.75)',
            'kernel': 'default (exponential)',
            'sample_around_instance': False,
            'num_features': 5,
            'num_samples': 5000,
            'sampling_func': 'gaussian',
            'sampling_func_params': {},
            'Stability (label = 1)': 1.0,
            'Mean R2 (label = 1)': 0.25875204620545944,
            'Stability (label = 2)': 1.0,
            'Mean R2 (label = 2)': 0.13010193899788192,
            'Stability (label = 0)': 1.0,
            'Mean R2 (label = 0)': 0.24258214084437704
        }

        for key, value in expected_results.items():
            if isinstance(value, float):
                self.assertAlmostEqual(results[key], value, places=7)
            else:
                self.assertEqual(results[key], value)

        # Test if other display methods work
        explanation = experiment._explanations[0]
        self.assertIsNotNone(explanation.as_map(), "as_map method returned None")
        self.assertIsNotNone(explanation.as_pyplot_figure(), "as_pyplot_figure method returned None")


if __name__ == '__main__':
    unittest.main()
