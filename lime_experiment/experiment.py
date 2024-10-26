import os
from datetime import datetime, timezone
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import RandomState

from common.name_utils import generate_short_uuid
from lime_experiment.models import LimeExperimentConfig, LabelExplanationMetrics
from lime.explanation import Explanation
from lime.lime_tabular import LimeTabularExplainer
from lime.metrics import calculate_stability

from common.generic import printc, pemji

RESULTS_OUTPUT_DIR = "./_experiment_results"
RESULTS_FILE_NAME = "experiment_results.csv"


class LimeExperiment:

    def __init__(self, config: LimeExperimentConfig):

        self._config = config
        self._explanations = None
        self._evaluation_results = None
        self._start_time = None
        self._end_time = None
        self._experiment_id = generate_short_uuid()

    def run(self):

        if self._end_time is not None:
            printc(f"{pemji('warning')} Experiment has already been run.", "r")
            return

        printc(
            f"{pemji('rocket')} Experiment started at {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]} UTC",
            "b")
        self._start_time = datetime.now(timezone.utc)

        printc(f"{pemji('gear')} Running the experiment...", "p")
        self._explanations = self._run_experiment(self._config.random_seed, self._config.experiment_data,
                                                  self._config.explainer_config, self._config.times_to_run,
                                                  self._config.explained_model, self._config.mode,
                                                  self._config.class_label_to_test)

        printc(f"{pemji('check_mark')} Experiment run complete. Starting evaluation of explanations...", "g")
        self._evaluation_results = self._evaluate_explanations(self._explanations)

        self._end_time = datetime.now(timezone.utc)
        printc(
            f"{pemji('check_mark')} Experiment completed at {self._end_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]} UTC",
            "g")

        # Calculate duration
        duration = self._end_time - self._start_time
        printc(f"{pemji('hourglass')} Experiment duration: {str(duration)}", "b")

        if self._config.save_results:
            printc(f"{pemji('floppy_disk')} Saving results...", "y")
            self._save_results()

    @staticmethod
    def _run_experiment(random_seed, experiment_data, explainer_config, times_to_run, explained_model, mode, class_label_to_test) -> List[
        Explanation]:

        if random_seed is not None:
            experiment_random_state = RandomState(random_seed)
            test_instance = experiment_data.get_random_test_instance(random_seed, class_label_to_test)
        else:
            experiment_random_state = None
            test_instance = experiment_data.get_random_test_instance(1, class_label_to_test)

        explainer = LimeTabularExplainer(
            training_data=experiment_data.get_training_data(),
            mode=mode,
            feature_names=experiment_data.get_feature_names(),
            categorical_features=experiment_data.get_categorical_features(),
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
        labels.sort()
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

        stability = np.mean([metrics.stability for metrics in self._evaluation_results.values()])
        fidelity = np.mean([metrics.fidelity for metrics in self._evaluation_results.values()])
        evaluation_results = {"Stability (average)": stability, "Mean R2 (average)": fidelity}
        for label, metrics in self._evaluation_results.items():
            evaluation_results.update({f"Stability (label = {label})": metrics.stability})
            evaluation_results.update({f"Mean R2 (label = {label})": metrics.fidelity})

        results = {"id": self._experiment_id, "start_time": self._start_time, "end_time": self._end_time,
                   "test_row_index": self._config.experiment_data.random_test_row_index,
                   "test_row_label": self._config.experiment_data.random_test_row_label}
        results.update(self._config.as_records_dict())
        results.update(evaluation_results)

        return results

    def _save_results(self):

        os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

        results = self.get_results()
        results_file_path = os.path.join(RESULTS_OUTPUT_DIR, RESULTS_FILE_NAME)

        # Convert current results to DataFrame
        results_df = pd.DataFrame([results])

        # Check if the results file already exists
        if os.path.exists(results_file_path):
            # Load the existing file to check for column names
            existing_df = pd.read_csv(results_file_path, nrows=0)  # Load only the header (column names)
            existing_columns = existing_df.columns.tolist()
            new_columns = results_df.columns.tolist()

            if existing_columns == new_columns:
                # Columns match, append the results
                results_df.to_csv(results_file_path, mode='a', header=False, index=False)
                print(f"Results successfully appended to {results_file_path}")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_file_path = os.path.join(RESULTS_OUTPUT_DIR, f"experiment_results_{timestamp}.csv")
                results_df.to_csv(new_file_path, mode='w', header=True, index=False)
                print(f"Column mismatch! Results saved to new file: {new_file_path}")
        else:
            results_df.to_csv(results_file_path, mode='w', header=True, index=False)
            print(f"Results file created and saved to {results_file_path}")

        print("Results: ")
        print(results_df.to_string(index=False))

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
