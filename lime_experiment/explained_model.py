import joblib
from sklearn.base import is_classifier


class ExplainedModel:
    """
    A model that can be explained by LIME, supporting only Scikit-learn probabilistic classifiers.
    """

    def __init__(self, model_path: str):
        """
        Load and validate the model.

        :param model_path: Path to the joblib serialized model file.
        """
        self._model_path = model_path
        self._model = joblib.load(model_path)
        self._validate_model_type()

    @property
    def model_type(self):
        """Return the class name of the loaded model."""
        return self._model.__class__.__name__

    @property
    def model_path(self):
        """Return the path to the loaded model."""
        return self._model_path

    def predict_probabilities(self, x):
        """Predict class probabilities for the input data."""
        return self._model.predict_proba(x)

    def _validate_model_type(self):
        """Ensure the loaded model is a classifier and supports probability predictions."""
        if not is_classifier(self._model):
            raise ValueError(f"{self.model_type} is not a classifier.")
        if not hasattr(self._model, 'predict_proba'):
            raise AttributeError(f"{self.model_type} does not support probability prediction.")
