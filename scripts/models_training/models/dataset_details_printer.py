from utils.print_utils import printc
from utils.print_utils import pemji


def print_dataset_details(train_x, val_x, test_x, class_names, categorical_features):
    print("Created dataset (train, val, test):", train_x.shape, val_x.shape, test_x.shape)
    print("Y labels:", class_names)

    # Feature names
    feature_names = train_x.columns

    # Categorical features information extraction
    categorical_columns_dict = {
        train_x.columns.get_loc(col): train_x[col].unique().tolist()
        for col in categorical_features if col in train_x
    }

    continuous_columns = [col for col in train_x.columns if col not in categorical_features]

    # Feature summary
    printc(f"{pemji('')} Total number of features: {len(train_x.columns)}\n", 'v')

    # Categorical columns with their indices, names, and categories
    printc(f"{pemji('')} Categorical columns ({len(categorical_columns_dict)}):", 'v')
    for col_index, categories in categorical_columns_dict.items():
        col_name = train_x.columns[int(col_index)]  # Ensure col_index is treated as integer
        print(f"  - Index: {col_index}, Name: {col_name}, Categories: {categories}")

    # Continuous columns with their indices and names
    printc(f"\n{pemji('')} Continuous columns ({len(continuous_columns)}):", 'v')
    for col in continuous_columns:
        col_index = train_x.columns.get_loc(col)
        print(f"  - Index: {col_index}, Name: {col}")