from distutils.command.config import config
from typing import List

from numpy.random import RandomState

from experiment_things.models import LimeExperimentConfig
from lime.explanation import Explanation
from lime.lime_tabular import LimeTabularExplainer

class LimeExperiment:

    def __init__(self, config: LimeExperimentConfig):
        self.config = config

    def run(self):
        explanations = self._run_experiment(self.config.random_seed, self.config.experiment_data, self.config.explainer_config,
                                            self.config.times_to_run, self.config.explained_model, self.config.mode)

        return explanations

    def _run_experiment(self, random_seed, experiment_data, explainer_config, times_to_run, explained_model, mode) -> List[Explanation]:

        if random_seed is not None:
            experiment_random_state = RandomState(random_seed)
        else:
            experiment_random_state = None

        test_instance = experiment_data.get_random_test_instance()

        explainer = LimeTabularExplainer(
            training_data=experiment_data.get_training_data(),
            mode=mode,
            feature_names=experiment_data.get_feature_names(),
            categorical_features=experiment_data.get_categorical_features(),
            categorical_names=experiment_data.get_categorical_names(),
            class_names=experiment_data.get_class_names(),
            random_state=experiment_random_state,
            kernel_width=explainer_config.kernel_width,
            kernel=explainer_config.kernel,
            sample_around_instance=explainer_config.sample_around_instance)

        explainer.set_sampling_func(explainer_config.sampling_func, **explainer_config.sampling_func_params)

        explanations = []

        for i in range(times_to_run):
            if random_seed is not None:
                experiment_random_state.seed(random_seed)
            explanation = explainer.explain_instance(data_row=test_instance,
                                                     predict_fn=explained_model.predict_probabilities,
                                                     top_labels=experiment_data.get_num_classes(),
                                                     num_features=explainer_config.num_features,
                                                     num_samples=explainer_config.num_samples)
            explanations.append(explanation)
        return explanations

    def _evaluate_explanations(explanations: List[Explanation]):
        print("Warning: evaluation not implemented")
        return {}

    def _save_results(results):
        """return a list of results config and evaluation"""
        print("Warning: saving results not implemented")
        return None
