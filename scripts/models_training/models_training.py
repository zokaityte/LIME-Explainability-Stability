import os
import sys
import itertools
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.impute import SimpleImputer

from models.dataset_details_printer import print_dataset_details
from models.logistic_regression import LogisticRegressionModel
from models.random_forest import RandomForestClassifierModel
from models.decision_tree import DecisionTreeClassifierModel
from models.knn import KNeighborsClassifierModel
from scripts.models_training.models.catboost_model import CatBoostClassifierModel
from scripts.models_training.models.xgboost_model import XGBoostClassifierModel

# From utils includes
from utils.print_utils import printc
from utils.print_utils import pemji

DATA_DIR = '../../data'
DATASET_DIR = 'big_data_zero_corr_enc'
MODELS_DIR = '../../model_checkpoints'


def load_data(print_details=False, encode_labels=True):
    train_data_path = os.path.join(DATA_DIR, DATASET_DIR, 'train.csv')
    val_data_path = os.path.join(DATA_DIR, DATASET_DIR, 'val.csv')
    test_data_path = os.path.join(DATA_DIR, DATASET_DIR, 'test.csv')

    # Load dataset
    train_data = pd.read_csv(train_data_path, engine="pyarrow")
    val_data = pd.read_csv(val_data_path, engine="pyarrow")
    test_data = pd.read_csv(test_data_path, engine="pyarrow")

    # Split features and labels
    train_x, train_y = train_data.drop(columns=['Label']), train_data['Label']
    train_labels = [str(label) for label in train_y.unique()]
    train_features = train_x.columns.tolist()
    val_x, val_y = val_data.drop(columns=['Label']), val_data['Label']
    test_x, test_y = test_data.drop(columns=['Label']), test_data['Label']

    # Replace infinities with NaN
    train_x.replace([np.inf, -np.inf], np.nan, inplace=True)
    val_x.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_x.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    train_x = imputer.fit_transform(train_x)
    val_x = imputer.transform(val_x)
    test_x = imputer.transform(test_x)
    printc(f"{pemji('')} Split X and Y!", 'b')

    if print_details:
        print_dataset_details(train_x, val_x, test_x, [], [])

    return train_labels, train_features, train_x, val_x, test_x, train_y, val_y, test_y


def merge_csv(csv_list, model_name):
    dataframes = []
    for file in csv_list:
        df = pd.read_csv(file)
        dataframes.append(df)

    folder = os.path.dirname(csv_list[0])
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df.to_csv(f'{folder}/{model_name}_results.csv', index=False)


def train_model(model_class, model_name, train_labels, train_features, save_dir,
                train_x, val_x, test_x, train_y, val_y, test_y, current_timestamp, hyperparameter_grid):
    csv_list = []

    # Generate all combinations of hyperparameters
    param_names = list(hyperparameter_grid.keys())
    param_values = list(hyperparameter_grid.values())

    for param_combination in itertools.product(*param_values):
        # Build a dictionary of current parameters
        params = dict(zip(param_names, param_combination))

        # Print parameters
        param_str = ', '.join(f"{k}: {v}" for k, v in params.items())
        printc(f"{pemji('hourglass')} Training {model_name.upper()} with parameters: {param_str}", 'b')

        # Instantiate model
        model = model_class(**params)

        # Generate output paths for saving results
        model.generate_output_path(save_dir)

        # Train the model
        model.train(train_x, train_y, val_x, val_y)
        printc(f"{pemji('check_mark')} Trained {model_name.upper()} with parameters: {param_str}", 'g')

        # Evaluate on test set
        model.evaluate(test_x, test_y, current_timestamp)
        csv_list.append(f"{model.test_output_path}.csv")

        # Evaluate on validation set
        model.evaluate(val_x, val_y, current_timestamp, is_test=False)
        csv_list.append(f"{model.val_output_path}.csv")

        # Save the model and tree image if applicable
        model.save()
        try:
            model.save_tree_image(train_labels, train_features, train_x, train_y)
        except Exception as e:
            printc(f"{pemji('warning')} Could not save tree image due to: {e}", 'r')
        printc(f"{pemji('download')} Saved {model_name.upper()} model with parameters: {param_str}; to: {save_dir}", 'g')

    # Merge all CSV files generated during evaluation
    merge_csv(csv_list, model_name)


if __name__ == '__main__':
    # Create missing directories
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    checkpoint_dir = f'{MODELS_DIR}/{DATASET_DIR}'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Load data
    train_labels, train_features, train_x_np, val_x_np, test_x_np, train_y, val_y, test_y = load_data(False, False)

    current_timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M')

    # Define hyperparameter grids for each model
    hyperparameter_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10],
        'max_features': ['sqrt', 'log2'],
        'random_state': [42],
        'n_jobs': [os.cpu_count()]
    }

    hyperparameter_grid_xgb = {
        'n_estimators': [100, 200],  # Number of trees
        'max_depth': [3, 6],  # Tree depth (controls complexity)
        'learning_rate': [0.05, 0.1],  # Step size shrinkage
        'subsample': [0.8],  # Row subsampling (stabilizes training)
        'colsample_bytree': [0.8],  # Feature subsampling
        'min_child_weight': [1, 3],  # Minimum sum of instance weights
        'use_label_encoder': [False],
    }
    hyperparameter_grid_catboost = {
        'iterations': [200, 300],
        'depth': [6, 8],
        'learning_rate': [0.1],
        'l2_leaf_reg': [3, 5],
        'bootstrap_type': ['Bernoulli'],
        'subsample': [0.8]
    }

    hyperparameter_grid_dt = {
        'random_state': [3],
        'max_depth': [3],
        'max_features': [3]
    }

    hyperparameter_grid_knn = {
        'n_neighbors': list(range(3, 7)),
        'weights': ['uniform'],  # Options: 'uniform', 'distance'
        'algorithm': ['auto'],   # Options: 'auto', 'ball_tree', 'kd_tree', 'brute'
        'n_jobs': [os.cpu_count()]
    }

    # Train models
    # Example for Random Forest
    # train_model(RandomForestClassifierModel, 'rf', train_labels, train_features, checkpoint_dir,
    #             train_x_np, val_x_np, test_x_np, train_y, val_y, test_y, current_timestamp, hyperparameter_grid_rf)

    # Example for XGBoost
    # train_model(XGBoostClassifierModel, 'xgboost', train_labels, train_features, checkpoint_dir,
    #             train_x_np, val_x_np, test_x_np, train_y, val_y, test_y, current_timestamp, hyperparameter_grid_xgb)

    # Example for CatBoost
    train_model(CatBoostClassifierModel, 'catboost', train_labels, train_features, checkpoint_dir,
                train_x_np, val_x_np, test_x_np, train_y, val_y, test_y, current_timestamp, hyperparameter_grid_catboost)

    # Example for Decision Tree
    # train_model(DecisionTreeClassifierModel, 'dt', train_labels, train_features, checkpoint_dir,
    #             train_x_np, val_x_np, test_x_np, train_y, val_y, test_y, current_timestamp, hyperparameter_grid_dt)

    # Example for K-Nearest Neighbors
    # train_model(KNeighborsClassifierModel, 'knn', train_labels, train_features, checkpoint_dir,
    #             train_x_np, val_x_np, test_x_np, train_y, val_y, test_y, current_timestamp, hyperparameter_grid_knn)
