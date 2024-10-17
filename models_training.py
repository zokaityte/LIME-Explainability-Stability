import os.path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from model.dataset_details_printer import print_dataset_details
from model.random_forest import RandomForestClassifierModel
from model.decision_tree import DecisionTreeClassifierModel
from model.regression import LinearRegressionModel
from model.knn import KNeighborsClassifierModel

DATA_DIR = 'data'
DATASET_DIR = 'sample_dataset_1'
MODELS_DIR = 'model_checkpoints'

CATEGORICAL_FEATURES = ["Dst Port", 'Timestamp', 'Label', 'Protocol', 'Fwd PSH Flags', 'FIN Flag Cnt', 'SYN Flag Cnt',
                        'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'ECE Flag Cnt',
                        'Fwd Seg Size Min']

# TODO Dropping these categorical features for now, need to handle them later. Leaving only binary categorical features.
COLUMNS_TO_DROP = [
    # 'Timestamp',  # Needs a way for enncoding (time of day?)
    # 'Dst Port',
    # # Categories: [53, 80, 22, 21, 3389, 64458, 443, 50158, 445, 50243, 49730, 51240, 0, 55146, 49942, 50197, 41096, 51579, 49154, 54429, 52737, 49948, 50574, 52147, 49681, 10884, 56754, 44285, 50667, 5355, 50576, 51311, 51090, 63467, 55232, 50633, 50516, 50122, 50746, 51740, 30158, 51902, 49713, 50227, 137, 54698, 49615, 51189, 52165, 50865, 13056, 49552, 51182, 21462, 49480, 49739, 51392, 25603, 49867, 49886, 60827, 51966, 54706, 49870, 50900, 52013, 50608, 50228, 51324, 51445, 50074, 2046, 50073, 55189, 63979, 50376, 38872, 4899, 49995, 50593, 62410]
    # 'Protocol',  # Categories: [17, 6, 0] (one-hot ?)
    # 'Fwd Seg Size Min'  # Categories: [8, 20, 32, 40, 0, 28] (one-hot ?)
]
# TODO^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# TODO remove when correct dataset is used
CATEGORICAL_FEATURES = [col for col in CATEGORICAL_FEATURES if col not in COLUMNS_TO_DROP]
# TODO^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def load_data(print_details=False, encode_labels=True):
    train_data_path = os.path.join(DATA_DIR, DATASET_DIR, 'train_data.csv')
    val_data_path = os.path.join(DATA_DIR, DATASET_DIR, 'val_data.csv')
    test_data_path = os.path.join(DATA_DIR, DATASET_DIR, 'test_data.csv')

    # Load dataset
    train_data = pd.read_csv(train_data_path).drop(columns=COLUMNS_TO_DROP)
    val_data = pd.read_csv(val_data_path).drop(columns=COLUMNS_TO_DROP)
    test_data = pd.read_csv(test_data_path).drop(columns=COLUMNS_TO_DROP)


    # TODO remove when correct dataset is used
    # Drop rows with infinity values
    def drop_infinity_rows(df):
        return df[~df.isin([np.inf, -np.inf]).any(axis=1)]
    # TODO^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # TODO remove when correct dataset is used
    # Apply the function to each dataset
    train_data = drop_infinity_rows(train_data)
    val_data = drop_infinity_rows(val_data)
    test_data = drop_infinity_rows(test_data)
    # TODO^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # Split features and labels
    train_x, train_y = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    val_x, val_y = val_data.iloc[:, :-1], val_data.iloc[:, -1]
    test_x, test_y = test_data.iloc[:, :-1], test_data.iloc[:, -1]

    if encode_labels:
        # Setup encoder for labels
        label_encoder = LabelEncoder()
        label_encoder.fit(pd.concat([train_y, val_y, test_y]))
        class_names = label_encoder.classes_

        # Encode labels
        train_y = label_encoder.transform(train_y)
        val_y = label_encoder.transform(val_y)
        test_y = label_encoder.transform(test_y)

    # Turn to numpy arrays
    train_x_np = train_x.to_numpy()
    val_x_np = val_x.to_numpy()
    test_x_np = test_x.to_numpy()

    if print_details:
        print_dataset_details(train_x, val_x, test_x, class_names, CATEGORICAL_FEATURES)

    return train_x_np, val_x_np, test_x_np, train_y, val_y, test_y



def train_random_forest(save_dir, train_x, val_x, test_x, train_y, val_y, test_y):
    for i in range(1,3):
        n_estimators=i
        random_state=i
        max_depth=i
        max_features=i

        model = RandomForestClassifierModel(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth, max_features=max_features)

        model.generate_output_path(save_dir)
        model.train(train_x, train_y)
        model.evaluate(val_x, val_y)
        model.save()


def train_regression(save_dir, train_x, val_x, test_x, train_y, val_y, test_y):
    model = LinearRegressionModel(fit_intercept=True)

    model.generate_output_path(save_dir)
    model.train(train_x, train_y)
    model.evaluate(val_x, val_y)
    model.save()


def train_decision_tree(save_dir, train_x, val_x, test_x, train_y, val_y, test_y):
    for i in range(3,7):
        random_state=i
        max_depth=i
        max_features=i

        model = DecisionTreeClassifierModel(random_state=random_state, max_depth=max_depth, max_features=max_features)

        model.generate_output_path(save_dir)
        model.train(train_x, train_y)
        model.evaluate(val_x, val_y)
        model.save()


def train_knn(save_dir, train_x, val_x, test_x, train_y, val_y, test_y):
    for i in range(3,7):
        n_neighbors=i
        weights='uniform' #{‘uniform’, ‘distance’}
        algorithm='auto' #{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default =’auto’

        model = KNeighborsClassifierModel(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)

        model.generate_output_path(save_dir)
        model.train(train_x, train_y)
        model.evaluate(val_x, val_y)
        model.save()



if __name__ == '__main__':
    # create missing dirs
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    checkpoint_dir = f'{MODELS_DIR}/{DATASET_DIR}'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # loda data
    train_x_np, val_x_np, test_x_np, train_y, val_y, test_y = load_data(False, True)

    # train model(s)

    # train_regression(checkpoint_dir, train_x_np, val_x_np, test_x_np, train_y, val_y, test_y)
    # train_decision_tree(checkpoint_dir, train_x_np, val_x_np, test_x_np, train_y, val_y, test_y)
    train_knn(checkpoint_dir, train_x_np, val_x_np, test_x_np, train_y, val_y, test_y)
    # train_random_forest(checkpoint_dir, train_x_np, val_x_np, test_x_np, train_y, val_y, test_y)