from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from lime.lime_tabular import LimeTabularExplainer


def test_lime_tabular_explainer_3_classes():
    # Generate classification data
    x_raw, y_raw = make_classification(n_classes=3, n_features=5, n_informative=3, n_samples=10, random_state=42)

    # Train classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(x_raw, y_raw)

    # Explain
    explainer = LimeTabularExplainer(x_raw, kernel_width=3, random_state=42, sample_around_instance=False)
    # TODO investigate the effect of sample_around_instance. Sample around instance is set to default False
    explanation = explainer.explain_instance(x_raw[0, :], clf.predict_proba, sampling_method='gaussian')

    # Test if explanation is correct
    assert explanation.as_list() == [('3', 0.043831946522315667), ('2', -0.043577987862162285),
                                     ('0', 0.03887729586806089), ('4', 0.019489294650031276),
                                     ('1', -0.014745938524001936)]

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
    explainer = LimeTabularExplainer(x_raw, kernel_width=3, random_state=42, sample_around_instance=False)
    explanation = explainer.explain_instance(x_raw[0, :], clf.predict_proba, sampling_method='gaussian')

    assert explanation.as_list() == [('0', 0.2233575658653036), ('1', 0.10089600235235874)]


if __name__ == '__main__':
    test_lime_tabular_explainer_3_classes()
    test_lime_tabular_explainer_2_classes()
