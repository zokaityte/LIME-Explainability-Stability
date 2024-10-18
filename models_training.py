import os.path
import sys

import numpy as np
import pandas as pd

from model.dataset_details_printer import print_dataset_details
from model.logistic_regression import LogisticRegressionModel
from model.random_forest import RandomForestClassifierModel
from model.decision_tree import DecisionTreeClassifierModel
from model.knn import KNeighborsClassifierModel
from sklearn.impute import SimpleImputer

from datetime import datetime

# From common includes
from common.generic import printc
from common.generic import pemji


DATA_DIR = 'data'
DATASET_DIR = 'sample_dataset_1' # or smoldata/bigdata
MODELS_DIR = 'model_checkpoints'


def load_data(print_details=False, encode_labels=True):
    train_data_path = os.path.join(DATA_DIR, DATASET_DIR, 'train_data.csv')
    val_data_path = os.path.join(DATA_DIR, DATASET_DIR, 'val_data.csv')
    test_data_path = os.path.join(DATA_DIR, DATASET_DIR, 'test_data.csv')

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

    # float64 limit into nan
    train_x.replace([np.inf, -np.inf], np.nan, inplace=True)
    val_x.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_x.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Nans as mean
    imputer = SimpleImputer(strategy='mean')
    train_x = imputer.fit_transform(train_x)
    val_x = imputer.transform(val_x)
    test_x = imputer.transform(test_x)
    printc(f"{pemji('')} Splitted X and Y!", 'b')

    if print_details:
        print_dataset_details(train_x, val_x, test_x, [], [])

    return train_labels, train_features, train_x, val_x, test_x, train_y, val_y, test_y


def merge_csv(csv_list, type):
    dataframes = []
    for file in csv_list:
        df = pd.read_csv(file)
        dataframes.append(df)
    
    folder = os.path.dirname(csv_list[0])

    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df.to_csv(f'{folder}/{type}_results.csv', index=False)


def train_random_forest(strain_labels, train_features, save_dir, train_x, val_x, test_x, train_y, val_y, test_y, current_timestamp):
    csv_list = []
    for i in range(3,7):
        n_estimators=i
        random_state=i
        max_depth=i
        max_features=i

        printc(f"{pemji('hourglass')} Training RF of parameters: random_state: {random_state}, max_depth: {max_depth}, max_features: {max_features}, n_estimators: {n_estimators}", 'b')
        model = RandomForestClassifierModel(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth, max_features=max_features, n_jobs=os.cpu_count())

        model.generate_output_path(save_dir)
        model.train(train_x, train_y)
        printc(f"{pemji('check_mark')} Training RF of parameters: random_state: {random_state}, max_depth: {max_depth}, max_features: {max_features}, n_estimators: {n_estimators}", 'g')

        # test evaluation
        model.evaluate(test_x, test_y, current_timestamp)
        csv_list.append(f"{model.test_output_path}.csv")

        # val evaluation
        model.evaluate(val_x, val_y, current_timestamp, is_test=False)
        csv_list.append(f"{model.val_output_path}.csv")

        model.save()
        # model.save_tree_image(strain_labels, train_features, train_x, train_y)
        printc(f"{pemji('download')} Saved RF of parameters: random_state: {random_state}, max_depth: {max_depth}, max_features: {max_features}, n_estimators: {n_estimators}; to: {save_dir}", 'g')
        
    merge_csv(csv_list, "rf")


def train_logistic_regression(save_dir, train_x, val_x, test_x, train_y, val_y, test_y, current_timestamp):
    csv_list = []
    printc(f"{pemji('hourglass')} Training regression", 'b')
    # penalty {‘l1’, ‘l2’, ‘elasticnet’, None}, default=’l2’
    model = LogisticRegressionModel(penalty='l2')

    model.generate_output_path(save_dir)
    model.train(train_x, train_y)
    printc(f"{pemji('check_mark')} Trained regression", 'g')

    # test evaluation
    model.evaluate(test_x, test_y, current_timestamp)
    csv_list.append(f"{model.test_output_path}.csv")

    # val evaluation
    model.evaluate(val_x, val_y, current_timestamp, is_test=False)
    csv_list.append(f"{model.val_output_path}.csv")

    model.save()
    printc(f"{pemji('download')} Saved regression", 'g')
    merge_csv(csv_list, "logreg")


def train_decision_tree(train_labels, train_features, save_dir, train_x, val_x, test_x, train_y, val_y, test_y, current_timestamp):
    csv_list = []
    for i in range(3,7):
        random_state=i
        max_depth=i
        max_features=i

        printc(f"{pemji('hourglass')} Training DF of parameters: random_state: {random_state}, max_depth: {max_depth}, max_features: {max_features}", 'b')
        model = DecisionTreeClassifierModel(random_state=random_state, max_depth=max_depth, max_features=max_features)

        model.generate_output_path(save_dir)
        model.train(train_x, train_y)
        printc(f"{pemji('check_mark')} Trained DF of parameters: random_state: {random_state}, max_depth: {max_depth}, max_features: {max_features}", 'g')

        # test evaluation
        model.evaluate(test_x, test_y, current_timestamp)
        csv_list.append(f"{model.test_output_path}.csv")

        # val evaluation
        model.evaluate(val_x, val_y, current_timestamp, is_test=False)
        csv_list.append(f"{model.val_output_path}.csv")
        model.save()

        # Save tree png
        # model.save_tree_image(train_labels, train_features, train_x, train_y)
        printc(f"{pemji('download')} Saved DF of parameters: random_state: {random_state}, max_depth: {max_depth}, max_features: {max_features}; to: {save_dir}", 'g')
        
    merge_csv(csv_list, "dt")


def train_knn(save_dir, train_x, val_x, test_x, train_y, val_y, test_y, current_timestamp):
    csv_list = []
    for i in range(3,7):
        n_neighbors=i
        weights='uniform' #{‘uniform’, ‘distance’}
        algorithm='auto' #{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default =’auto’

        printc(f"{pemji('hourglass')} Training KNN of parameters: n_neighbors: {n_neighbors}, weights: {weights}, algorithm: {algorithm}", 'b')
        model = KNeighborsClassifierModel(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, n_jobs=os.cpu_count())

        model.generate_output_path(save_dir)
        model.train(train_x, train_y)
        printc(f"{pemji('check_mark')} Trained KNN of parameters: n_neighbors: {n_neighbors}, weights: {weights}, algorithm: {algorithm}", 'g')

        # test evaluation
        model.evaluate(test_x, test_y, current_timestamp)
        csv_list.append(f"{model.test_output_path}.csv")

        # val evaluation
        model.evaluate(val_x, val_y, current_timestamp, is_test=False)
        csv_list.append(f"{model.val_output_path}.csv")

        model.save()
        printc(f"{pemji('download')} Saved KNN of parameters: n_neighbors: {n_neighbors}, weights: {weights}, algorithm: {algorithm}; to: {save_dir}", 'g')
        
    merge_csv(csv_list, "knn")



if __name__ == '__main__':
    # create missing dirs
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    checkpoint_dir = f'{MODELS_DIR}/{DATASET_DIR}'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # loda data
    train_labels, train_features, train_x_np, val_x_np, test_x_np, train_y, val_y, test_y = load_data(False, True)
    
    current_timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M')

    # train model(s)
    # train_logistic_regression(checkpoint_dir, train_x_np, val_x_np, test_x_np, train_y, val_y, test_y, current_timestamp)
    # train_decision_tree(train_labels, train_features, checkpoint_dir, train_x_np, val_x_np, test_x_np, train_y, val_y, test_y, current_timestamp)
    train_random_forest(train_labels, train_features, checkpoint_dir, train_x_np, val_x_np, test_x_np, train_y, val_y, test_y, current_timestamp)
    # train_knn(checkpoint_dir, train_x_np, val_x_np, test_x_np, train_y, val_y, test_y, current_timestamp)
