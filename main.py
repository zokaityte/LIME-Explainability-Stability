from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from experiment_things.experiment import LimeExperiment
from experiment_things.experiment_data import ExperimentData
from experiment_things.explained_model import ExplainedModel
from experiment_things.models import LimeExplainerConfig, LimeExperimentConfig
from lime.lime_tabular import LimeTabularExplainer


def test_lime_tabular_explainer_3_classes():
    # Generate classification data
    x_raw, y_raw = make_classification(n_classes=3, n_features=5, n_informative=3, n_samples=10, random_state=42)

    # Train classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(x_raw, y_raw)

    # Explain
    explainer = LimeTabularExplainer(x_raw, kernel_width=3, random_state=42, sample_around_instance=False)
    explanation = explainer.explain_instance(x_raw[0, :], clf.predict_proba)

    # Test if explanation is correct for the first class
    assert explanation.as_list(1) == [('3', 0.043831946522315667), ('2', -0.043577987862162285),
                                      ('0', 0.03887729586806089), ('4', 0.019489294650031276),
                                      ('1', -0.014745938524001936)]

    # Test if other display methods work
    explanation.as_map()
    explanation.as_pyplot_figure()

def test_multiple_runs():

    # SETUP
    random_seed = 42
    explained_model = ExplainedModel("model_checkpoints/test_rf_model.pkl")
    experiment_data = ExperimentData("data/sample_dataset_2/train.csv",
                                     "data/sample_dataset_2/val.csv",
                                     "data/sample_dataset_2/test.csv")
    explainer_config = LimeExplainerConfig(**{
        'kernel_width': 3,  # Default: sqrt(num_features) * 0.75 if None
        'kernel': None,  # Default: exponential kernel function if None
        'sample_around_instance': False,  # Default: False, sample features around instance
        'num_features': 10,  # Default: 10, maximum number of features in explanation
        'num_samples': 5000,  # Default: 5000, size of the neighborhood to generate perturbed samples
        'sampling_func': 'gaussian',
        'sampling_func_params': {}
    })
    times_to_run = 10

    # RUN
    experiment_config = LimeExperimentConfig(explained_model, experiment_data, explainer_config, times_to_run, random_seed, "classification")
    experiment = LimeExperiment(experiment_config)
    explanations = experiment.run()

    # Test if explanation is correct for the first class
    for explanation in explanations:
        assert explanation.as_list(1) == [('3', 0.04383194652231568), ('2', -0.04357798786216229), ('0', 0.03887729586806089), ('4', 0.019489294650031266), ('1', -0.01474593852400193)]

    # Test if other display methods work
    explanation.as_map()
    explanation.as_pyplot_figure()


def test_lime_tabular_explainer_2_classes():
    # Generate classification data
    x_raw, y_raw = make_classification(n_classes=2, n_features=2, n_informative=2, n_repeated=0, n_redundant=0,
                                       n_samples=10, random_state=42)

    # Train classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(x_raw, y_raw)

    # Explain
    explainer = LimeTabularExplainer(x_raw, kernel_width=3, random_state=42, sample_around_instance=True)
    num_samples = 1000

    # explainer.set_sampling_func('alpha', alpha=0.5, beta=1, location=5, scale=1)
    # explanation = explainer.explain_instance(x_raw[0, :], clf.predict_proba, num_samples=num_samples)
    # print(f'explanation {explanation.as_list()}')

    explainer.set_sampling_func('beta', alpha=0.7, beta_param=0.5, location=0, scale=1)
    explanation = explainer.explain_instance(x_raw[0, :], clf.predict_proba, num_samples=num_samples)
    print(f'explanation {explanation.as_list()}')

    # explainer.set_sampling_func('gamma', shape_param=7.0, scale=1.0)
    # explanation = explainer.explain_instance(x_raw[0, :], clf.predict_proba, num_samples=num_samples)
    # print(f'explanation {explanation.as_list()}')

    # explainer.set_sampling_func('pareto', shape_param=2.5, scale=1)
    # explanation = explainer.explain_instance(x_raw[0, :], clf.predict_proba, num_samples=num_samples)
    # print(f'explanation {explanation.as_list()}')

    # explainer.set_sampling_func('weibull', shape_param=4, scale=1)
    # explanation = explainer.explain_instance(x_raw[0, :], clf.predict_proba, num_samples=num_samples)
    # print(f'explanation {explanation.as_list()}')

    explainer.set_sampling_func('gaussian')
    explanation = explainer.explain_instance(x_raw[0, :], clf.predict_proba, num_samples=num_samples)
    print(f'explanation {explanation.as_list()}')
    # assert explanation.as_list() == [('0', 0.2233575658653036), ('1', 0.10089600235235874)]


if __name__ == '__main__':
    test_lime_tabular_explainer_3_classes()
    test_lime_tabular_explainer_2_classes()
    test_multiple_runs()
