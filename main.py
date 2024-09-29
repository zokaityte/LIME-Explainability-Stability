# Testing script

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer


def test_lime_tabular_explainer():

    # Generate classification data
    x_raw, y_raw = make_classification(n_classes=3, n_features=5, n_informative=3, n_samples=10, random_state=42)

    # Train classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(x_raw, y_raw)

    # Initialize LimeTabularExplainer
    explainer = LimeTabularExplainer(x_raw, discretize_continuous=False, kernel_width=3, random_state=42)

    # Explain single instance
    explanation = explainer.explain_instance(x_raw[0, :], clf.predict_proba)

    # Feature impact
    feature_importance = explanation.as_list()

    assert feature_importance == [('3', 0.043831946522315667), ('2', -0.043577987862162285), ('0', 0.03887729586806089), ('4', 0.019489294650031276), ('1', -0.014745938524001936)]



if __name__ == '__main__':
    test_lime_tabular_explainer()