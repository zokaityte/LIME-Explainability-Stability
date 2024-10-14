import unittest
from lime_experiment.experiment import LimeExperiment
from lime_experiment.experiment_data import ExperimentData
from lime_experiment.explained_model import ExplainedModel
from lime_experiment.models import LimeExplainerConfig, LimeExperimentConfig

class TestLimeExperiment(unittest.TestCase):

    def setUp(self):
        """Setup common data for the test."""
        self.random_seed = 42  # Or None
        self.explained_model = ExplainedModel("../model_checkpoints/test_rf_model.pkl")
        self.experiment_data = ExperimentData(
            "../datasets/sample_dataset_2",
            label_names=["0labelname", "1labelname", "2labelname"]
        )

        self.explainer_config = LimeExplainerConfig(**{
            'kernel_width': 3,
            'kernel': None,
            'sample_around_instance': False,
            'num_features': 10,
            'num_samples': 5000,
            'sampling_func': 'gaussian',
            'sampling_func_params': {}
        })

        self.times_to_run = 10

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
        expected_explanation = [('feature_2', -0.042447470132049844),
                                ('feature_0', 0.03661369437283125),
                                ('feature_3', 0.03275282434566029),
                                ('feature_4', 0.025438416754464212),
                                ('feature_1', -0.017154380819803368)]

        # Verify explanations
        for explanation in experiment._explanations:
            actual_explanation = explanation.as_list(1)
            print(actual_explanation)
            for i, (feature, value) in enumerate(expected_explanation):
                with self.subTest(i=i):
                    self.assertEqual(actual_explanation[i][0], feature)
                    self.assertAlmostEqual(actual_explanation[i][1], value, places=7)

        # Verify experiment results
        results = experiment.get_results()
        print(results)
        expected_results = {
            'dataset': '../datasets/sample_dataset_2',
            'number_of features': 10,
            'number_of_categorical_features': 0,
            'explained_model': '../model_checkpoints/test_rf_model.pkl',
            'explained_model_type': 'RandomForestClassifier',
            'times_explained': 10,
            'random_seed': 42,
            'kernel_width': 3,
            'kernel': 'default (exponential)',
            'sample_around_instance': False,
            'num_features': 10,
            'num_samples': 5000,
            'sampling_func': 'gaussian',
            'sampling_func_params': {},
            'Stability (label = 1)': 1.0,
            'Mean R2 (label = 1)': 0.2835911042919463,
            'Stability (label = 2)': 1.0,
            'Mean R2 (label = 2)': 0.706446360709742,
            'Stability (label = 0)': 1.0,
            'Mean R2 (label = 0)': 0.7376951927688562
        }

        for key, value in expected_results.items():
            if isinstance(value, float):
                self.assertAlmostEqual(results[key], value, places=7)
            else:
                self.assertEqual(results[key], value)

    def test_explanation_display_methods(self):
        """Test the display methods for explanations."""
        # RUN
        experiment_config = LimeExperimentConfig(
            self.explained_model, self.experiment_data, self.explainer_config,
            self.times_to_run, self.random_seed, save_explanations=False, save_results=False
        )
        experiment = LimeExperiment(experiment_config)
        experiment.run()

        # Test if other display methods work
        explanation = experiment._explanations[0]
        self.assertIsNotNone(explanation.as_map(), "as_map method returned None")
        self.assertIsNotNone(explanation.as_pyplot_figure(), "as_pyplot_figure method returned None")


if __name__ == '__main__':
    unittest.main()
