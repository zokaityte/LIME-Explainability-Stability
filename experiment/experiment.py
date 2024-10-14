import os
from datetime import datetime, timezone
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import RandomState

from common.name_utils import generate_slug_with_seed
from experiment.models import LimeExperimentConfig, LabelExplanationMetrics
from lime.explanation import Explanation
from lime.lime_tabular import LimeTabularExplainer
from lime.metrics import calculate_stability

RESULTS_OUTPUT_DIR = "./_experiment_results"
RESULTS_FILE_NAME = "experiment_results.csv"


class LimeExperiment:

    def __init__(self, config: LimeExperimentConfig):

        self._config = config
        self._explanations = None
        self._evaluation_results = None
        self._start_time = None
        self._end_time = None
        self._experiment_id = generate_slug_with_seed(config.random_seed)

    def run(self):

        if self._end_time is not None:
            print("Experiment has already been run.")
            return

        print(f"Experiment started at {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]} UTC")
        self._start_time = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

        print("Running the experiment...")
        self._explanations = self._run_experiment(self._config.random_seed, self._config.experiment_data,
                                                  self._config.explainer_config, self._config.times_to_run,
                                                  self._config.explained_model, self._config.mode)

        print("Experiment run complete. Starting evaluation of explanations...")
        self._evaluation_results = self._evaluate_explanations(self._explanations)

        self._end_time = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        print(f"Experiment completed at {self._end_time} UTC")

        if self._config.save_results:
            print("Saving results...")
            self._save_results()

    @staticmethod
    def _run_experiment(random_seed, experiment_data, explainer_config, times_to_run, explained_model, mode) -> List[
        Explanation]:

        if random_seed is not None:
            experiment_random_state = RandomState(random_seed)
        else:
            experiment_random_state = None

        test_instance = experiment_data.get_random_test_instance(random_seed)

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
            print(f"Running iteration {i + 1}/{times_to_run}...")
            if random_seed is not None:
                experiment_random_state.seed(random_seed)
            explanation = explainer.explain_instance(data_row=test_instance,
                                                     predict_fn=explained_model.predict_probabilities,
                                                     top_labels=experiment_data.get_num_classes(),
                                                     num_features=explainer_config.num_features,
                                                     num_samples=explainer_config.num_samples)
            explanations.append(explanation)

        print(f"Completed {times_to_run} iterations.")
        return explanations

    @staticmethod
    def _evaluate_explanations(explanations: List[Explanation]) -> Dict[str, LabelExplanationMetrics]:
        """
        Evaluates the explanations for all labels, calculating fidelity and stability metrics.

        Args:
        explanations (List[Explanation]): A list of Explanation objects to evaluate.

        Returns:
        Dict[str, LabelExplanationMetrics]: A dictionary where each label has its corresponding fidelity
                                            and stability metrics stored in a LabelExplanationMetrics dataclass.
        """
        labels = explanations[0].available_labels()
        results = {}

        for label in labels:
            # Collect features across runs for the current label
            features_across_runs = [[feature for feature, _ in explanation.as_list(label)] for explanation in
                                    explanations]

            # Calculate fidelity and stability metrics for the current label
            fidelity_metric = np.mean([explanation.score[label] for explanation in explanations])
            stability_metric = calculate_stability(features_across_runs)

            # Store the results using the LabelExplanationMetrics dataclass
            results[label] = LabelExplanationMetrics(fidelity=fidelity_metric, stability=stability_metric)
            print(f"Label {label} evaluated: Fidelity = {fidelity_metric}, Stability = {stability_metric}")

        return results

    def get_results(self):

        if not self._end_time:
            print("Experiment has not been run yet.")
            return

        evaluation_results = {}
        for label, metrics in self._evaluation_results.items():
            evaluation_results.update({f"Stability (label = {label})": metrics.stability})
            evaluation_results.update({f"Mean R2 (label = {label})": metrics.fidelity})

        results = {"id": self._experiment_id, "start_time": self._start_time, "end_time": self._end_time}
        results.update(self._config.as_records_dict())
        results.update(evaluation_results)

        return results

    def _save_results(self):

        os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

        results = self.get_results()
        results_file_path = os.path.join(RESULTS_OUTPUT_DIR, RESULTS_FILE_NAME)

        if os.path.exists(results_file_path):
            pd.DataFrame([results]).to_csv(results_file_path, mode='a', header=False, index=False)
        else:
            pd.DataFrame([results]).to_csv(results_file_path, mode='w', header=True, index=False)

        if os.path.exists(results_file_path):
            print(f"Results successfully saved to {results_file_path}")
        else:
            print(f"Error: Failed to save results to {results_file_path}")

        if self._config.save_explanations:
            experiment_path = os.path.join(RESULTS_OUTPUT_DIR, self._experiment_id)
            os.makedirs(experiment_path, exist_ok=True)
            for i, explanation in enumerate(self._explanations):
                for label in explanation.available_labels():
                    with open(os.path.join(experiment_path, f"explanation_{i}_{label}.txt"), "w") as f:
                        f.write(str(explanation.as_list(label)))
                    explanation.as_pyplot_figure(label).savefig(
                        os.path.join(experiment_path, f"explanation_{i}_{label}.png"))
                    plt.close()
