import json
from tqdm import tqdm
from lime_experiment.experiment import LimeExperiment
from lime_experiment.experiment_data import ExperimentData
from lime_experiment.explained_model import ExplainedModel
from lime_experiment.models import LimeExplainerConfig, LimeExperimentConfig


def run_experiment(config: dict):
    """Runs a single experiment with the provided dictionary configuration."""

    # Load the explained model
    explained_model = ExplainedModel(config['model_path'])

    # Load the dataset
    experiment_data = ExperimentData(config['dataset_path'], label_names=config['label_names'],
                                     categorical_columns_names=config.get('categorical_columns_names', None))

    # Setup LIME explainer configuration
    lime_explainer_config = LimeExplainerConfig(
        kernel_width=config['explainer_config']['kernel_width'],
        kernel=config['explainer_config']['kernel'],
        sample_around_instance=config['explainer_config']['sample_around_instance'],
        num_features=config['explainer_config']['num_features'],
        num_samples=config['explainer_config']['num_samples'],
        sampling_func=config['explainer_config']['sampling_func'],
        sampling_func_params=config['explainer_config']['sampling_func_params']
    )

    # Setup the experiment configuration for LIME
    lime_experiment_config = LimeExperimentConfig(
        explained_model=explained_model,
        experiment_data=experiment_data,
        explainer_config=lime_explainer_config,
        times_to_run=config['times_to_run'],
        random_seed=config.get('random_seed', None),
        class_label_to_test=config['class_label_to_explain'])

    experiment = LimeExperiment(config=lime_experiment_config)
    experiment.run()


def load_and_run_experiments(json_file: str):
    """Loads experiments from a JSON file and runs each experiment."""

    with open(json_file, 'r') as f:
        experiment_configs = json.load(f)['experiments']

    with tqdm(total=len(experiment_configs), desc="Running Experiments", unit="experiment") as pbar:
        for config in experiment_configs:
            run_experiment(config)
            pbar.update(1)


if __name__ == "__main__":
    json_file = 'experiments_config.json'
    load_and_run_experiments(json_file)

