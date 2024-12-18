import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Function to load data and calculate feature variability
def process_explanations(file_path):
    data = pd.read_csv(file_path)

    # Extract feature importance values
    def extract_feature_importances(results):
        features = {}
        for item in results.strip("[]").split("), ("):
            feature, importance = item.split(",")
            features[feature.strip("('")] = float(importance.strip(")"))
        return features

    data['feature_importances'] = data['results'].apply(extract_feature_importances)

    # Convert to matrix form (runs x features)
    importance_matrix = pd.DataFrame(data['feature_importances'].tolist()).fillna(0)

    # Calculate standard deviation for each feature
    feature_variability = importance_matrix.std(axis=0)

    return feature_variability


# Replace with your actual file paths
file1_path = '../../explanations_gaussian_dd8a5065.csv'  # Gaussian experiment file
file2_path = '../../explanations_pareto_f1719849.csv'  # Non-Gaussian experiment file

# Process both files
feature_variability_1 = process_explanations(file1_path)
feature_variability_2 = process_explanations(file2_path)

# Plot histograms for both datasets
plt.figure(figsize=(14, 6))

# Plot for Gaussian
plt.subplot(1, 2, 1)
plt.hist(feature_variability_1, bins=10, color='skyblue', edgecolor='black')
plt.title('Feature Importance Variability (Gaussian)')
plt.xlabel('Standard Deviation of Feature Importance')
plt.ylabel('Number of Features')
plt.grid(axis='y')

# Plot for Non-Gaussian
plt.subplot(1, 2, 2)
plt.hist(feature_variability_2, bins=10, color='orange', edgecolor='black')
plt.title('Feature Importance Variability (Non-Gaussian)')
plt.xlabel('Standard Deviation of Feature Importance')
plt.ylabel('Number of Features')
plt.grid(axis='y')

plt.tight_layout()
plt.show()
