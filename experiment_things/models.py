from dataclasses import dataclass

from experiment_things.explained_model import ExplainedModel
from experiment_things.experiment_data import ExperimentData


@dataclass
class LimeExplainerConfig:
    kernel_width: int
    kernel: callable
    sample_around_instance: bool
    num_features: int
    num_samples: int
    sampling_func: str
    sampling_func_params: dict
    """
    In these experiments, the following explainer constants will be maintained: the model_regressor will default to Ridge 
    regression. The explanation will be generated for all labels of the model. The 
    Euclidean distance metric will be used as the default for calculating proximity in feature space, which impacts 
    how the local models are weighted. Additionally, the feature selection method will default to 'auto', 
    allowing LIME to automatically choose the best features based on criteria like forward selection or Lasso, 
    depending on the data and model.
    """


@dataclass
class LimeExperimentConfig:
    explained_model: ExplainedModel
    experiment_data: ExperimentData
    explainer_config: LimeExplainerConfig
    times_to_run: int
    random_seed: int
    mode: str
