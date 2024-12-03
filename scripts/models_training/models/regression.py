import csv
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

from common.generic import printc
from common.generic import pemji


class LinearRegressionModel:
    def __init__(self, *args, **kwargs):
        self.output_path = None
        self.training_params = kwargs
        self.model = LinearRegression(*args, **kwargs)

    def generate_output_path(self, path):
        # Start the output path with the base path
        output_path = f'{path}/lr/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        print(self.training_params)
        # Iterate over the kwargs and append them to the output path in the specified format
        for key, value in self.training_params.items():
            output_path += f'{key}_{value}'

        self.output_path = output_path

    def train(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def evaluate(self, test_x, test_y, current_timestamp):
        # Get the predictions
        y_pred = self.model.predict(test_x)

        # Calculate metrics
        mse = mean_squared_error(test_y, y_pred)
        r2 = r2_score(test_y, y_pred)

        # Export metrics to CSV
        self.export_metrics_to_csv(f'{self.output_path}.csv', mse, r2, current_timestamp)

        printc(f"{pemji('rocket')} Trained Regression metrics: MSE: {mse}, R-Squared: {r2}", 'v')

        # Return all the metrics for further use
        return mse, r2

    def export_metrics_to_csv(self, filename, mse, r2, current_timestamp):
        # Convert the training params to a string format
        params_str = '/'.join(f'{key}={value}' for key, value in self.training_params.items())
        # Create a list of metrics
        data = [
            ["Timestamp", "Training params", "Mean Squared Error", "R-squared"],
            [current_timestamp, params_str, mse, r2]
        ]

        # Export to CSV
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(data)

    def save(self):
        joblib.dump(self.model, f'{self.output_path}.pkl')

    def load(self, path):
        self.model = joblib.load(path)