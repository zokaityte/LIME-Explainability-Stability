import os
import sys
import pandas as pd
import gc
import plotly.express as px
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import rankdata
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

# Include path for generics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

# From common includes
from generic import printc
from generic import pemji


def __spearman_corr_fast(x, y):
    if np.isnan(x).any() or np.isnan(y).any():
        # Mask NaNs before ranking
        mask = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[mask], y[mask]

    # Rank the data, handling NaNs by omitting them
    ranked_x = rankdata(x, method='average')
    ranked_y = rankdata(y, method='average')

    n = len(ranked_x)
    if n == 0:
        return np.nan, np.nan  # return NaN if no valid data
    
    # Calculate covariance between ranks
    cov = np.cov(ranked_x, ranked_y)[0, 1]
    
    # Calculate standard deviations
    std_x = np.std(ranked_x, ddof=1)
    std_y = np.std(ranked_y, ddof=1)
    
    # Spearman correlation coefficient
    spearman_corr = cov / (std_x * std_y)
    
    # Calculate p-value based on Spearman correlation
    t = spearman_corr * np.sqrt((n - 2) / (1 - spearman_corr**2))
    p_value = 2 * (1 - norm.cdf(abs(t)))

    return spearman_corr, p_value


def fast_spearman(data):
    """
    Fast calculation of spearman correlation coefficient with p values using parallel processing.

    Parameters:
    data (dataframe): input df data

    Returns:
    corr_df (dataframe): spearman correlation coefficients of data
    p_values_df (dataframe): p values of spearman correlations
    """

    cpu_count = int(os.cpu_count() * 0.8)
    printc(f"{pemji('rocket')} Calculating spearman now! CPU cnt will use: {cpu_count}", 'b')
    n_cols = data.shape[1]
    corr_matrix = np.zeros((n_cols, n_cols))
    p_matrix = np.zeros((n_cols, n_cols))
    completed_tasks = 0
    total_tasks = (n_cols * (n_cols + 1)) // 2

    # Worker function to compute the Spearman correlation and p-value for a given pair of columns
    def worker(i, j):
        corr, p_val = __spearman_corr_fast(data.iloc[:, i], data.iloc[:, j])
        return i, j, corr, p_val

    # List to store the future objects
    results = []

    # Parallelize with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_indices = {
            executor.submit(worker, i, j): (i, j) for i in range(n_cols) for j in range(i, n_cols)
        }

        # Collect the results as the tasks complete
        for future in as_completed(future_to_indices):
            i, j = future_to_indices[future]
            completed_tasks += 1
            col_i_name = data.columns[i]
            col_j_name = data.columns[j]
            printc(f"{pemji('lightning')} Completed: {col_i_name} vs {col_j_name} ({completed_tasks}/{total_tasks} tasks done)", 'b')
            i, j, corr, p_val = future.result()
            results.append((i, j, corr, p_val))

    # Write the results back to the matrices in the correct order
    for i, j, corr, p_val in sorted(results, key=lambda x: (x[0], x[1])):
        corr_matrix[i, j] = corr_matrix[j, i] = corr
        p_matrix[i, j] = p_matrix[j, i] = p_val

    # Convert the matrices to DataFrames
    corr_df = pd.DataFrame(corr_matrix, index=data.columns, columns=data.columns)
    p_values_df = pd.DataFrame(p_matrix, index=data.columns, columns=data.columns)

    return corr_df, p_values_df


class bigdata:
    def __init__(self, csv_path="data.csv", load_splits=False, splitted_dir='splitted', train_file="train.csv", val_file="val.csv", test_file="test.csv"):
        """
        Create a big data object to handle big data efficiently and fast.
        Regular methods for statistics, plots are slow and managing memory with big data is cumbersome.

        Parameters:
        csv_path (path): path for main data csv
        load_splits (bool): if we use splitted data csv, then load them instead of whole main csv
        splitted_dir (path): splitted data csv's dir
        train_file (string): splitted train csv filename in splitted_dir
        val_file (string): splitted val csv filename in splitted_dir
        test_file (string): splitted test csv filename in splitted_dir

        Returns:
        class big_data: object
        """
        
        # Do not access directly!
        if load_splits is False:
            self.__data = pd.read_csv(csv_path, engine="pyarrow") #Faster!
            if self.__data is None:
                raise ValueError(f"{pemji('red_cross')} CSV data read failure! NULL!")
        
            self.data_freed = False # We have whole csv data in memory as one object
            self.__train_df = None
            self.__test_df = None
            self.__val_df = None
        else:
            self.__data = None
            self.data_freed = True # No big main csv data
            self.load_splits(splitted_dir, train_file, val_file, test_file)
        
        # TODO: load if provided here or elsewhere
        self.main_corr_df = None
        self.main_label_corr_df = None
        self.train_corr_df = None
        self.train_label_corr_df = None
        self.val_corr_df = None
        self.val_label_corr_df = None
        self.test_corr_df = None
        self.test_label_corr_df = None

        printc(f"{pemji('check_mark')} Inited csv data from {csv_path}", 'g')

    def get_main_data(self):
        """
        Return main csv dataframe safely.

        Parameters:
        None

        Returns:
        pd.Dataframe: data
        """
        
        if self.data_freed is True:
            raise ValueError(f"{pemji('red_cross')}{pemji('')} Data was already freed! You maybe wanted to use train/val/test obj?")
        elif self.__data is not None:
            return self.__data
        else:
            raise ValueError(f"{pemji('red_cross')} Data is either invalid or not set")
        
    def free_main_data(self):
        """
        Free main not-splitted csv object data when done using main object or splitted.
        Please use this when working with big data to conserve RAM.

        Parameters:
        None

        Returns:
        None
        """
        
        if self.data_freed == False:
            self.data_freed = True
            del self.__data
            gc.collect()
        else:
            # We call free when we did something with data. So what kind of NULL were you working with before this...
            raise ValueError(f"{pemji('red_cross')} Invalid use of free on already NULL data! What have you done?{pemji('red_exclamation')}")

    def get_data(self):
        """
        Return splitted dataframes safely.

        Parameters:
        None

        Returns:
        pd.Dataframe: train, val, test
        """
        
        if self.__train_df is None or self.__val_df is None or self.__test_df is None:
            raise ValueError(f"{pemji('red_cross')}{pemji('')} Train/Val/Test is either invalid or not set! Not yet split()?")
        return self.__train_df, self.__val_df, self.__test_df
    
    def transform_timestamp(self, timestamp_col: str) -> pd.DataFrame:
        """
        Transforms the Timestamp column to show only the hour as an integer.

        Parameters:
        timestamp_col (str): The name of the timestamp column.

        Returns:
        Nothing, updates the data inside Class object only!
        """
        
        if self.data_freed == False:
            df_list = [self.__data]
        elif self.__train_df and self.__val_df and self.__test_df:
            df_list = [self.__train_df, self.__val_df, self.__test_df]
        else:
            raise ValueError(f"{pemji('red_cross')} No data to timestamp encode in object!")

        for df in df_list:
            # Convert the column to datetime format
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='%d/%m/%Y %H:%M:%S')
            
            # Extract the hour and replace the timestamp column with it
            df[timestamp_col] = df[timestamp_col].dt.hour
            
        printc(f"{pemji('green_check')} Timestamp encoding done on columns: {timestamp_col}", 'g')
    
    def one_hot_encode(self, categorical_cols: list) -> pd.DataFrame:
        """
        One-hot encodes the specified categorical columns in the dataframe.
        
        Parameters:
        categorical_cols (list): List of categorical column names to be one-hot encoded.
        
        Returns:
        Nothing, updates the data inside Class object only!
        """
        
        if self.data_freed == False:
            self.__data = pd.get_dummies(self.__data, columns=categorical_cols)
        elif self.__train_df and self.__val_df and self.__test_df:
            self.__train_df = pd.get_dummies(self.__train_df, columns=categorical_cols)
            self.__val_df = pd.get_dummies(self.__val_df, columns=categorical_cols)
            self.__test_df = pd.get_dummies(self.__test_df, columns=categorical_cols)
        else:
            raise ValueError(f"{pemji('red_cross')} No data to one-hot encode in object!")
        
        printc(f"{pemji('green_check')} One-hot encoding done on columns: {categorical_cols}", 'g')


    def conditional_prob_transform(self, categorical_cols: list, target_col: str, alpha: float = 10):
        """
        Applies smoothed conditional probability-based transformation (target encoding) to the specified categorical columns.
        
        Parameters:
        categorical_cols (list): List of categorical column names to be transformed.
        target_col (str): The name of the target column for conditional probability transformation.
        alpha (float): Smoothing factor. Higher values of alpha give more weight to the global mean.
        
        Returns:
        Nothing, updates the data inside Class object only!
        """

        if self.data_freed == False:
            df_list = [self.__data]
        elif self.__train_df and self.__val_df and self.__test_df:
            df_list = [self.__train_df, self.__val_df, self.__test_df]
        else:
            raise ValueError(f"{pemji('red_cross')} No data to cond prob encode in object!")

        mean_list = []
        for df in df_list:
            mean_list.append(df[target_col].mean())

        for col in categorical_cols:
            for idx, df in enumerate(df_list):
                # Calculate category means and sizes
                category_means = df.groupby(col)[target_col].mean()
                category_sizes = df.groupby(col).size()

                # Apply smoothing
                smoothed = (category_sizes * category_means + alpha * mean_list[idx]) / (category_sizes + alpha)

                # Replace categorical values with their corresponding smoothed probabilities
                df[col] = df[col].map(smoothed)

        printc(f"{pemji('green_check')} Conditional probability transform done on columns: {categorical_cols} vs {target_col}", 'g')


    def split(self, output_dir="splitted", y_labels=['Label'], free_whole=True, train_split=0.7, val_split=0.2, test_split=0.1, seed=42):
        """
        Split main data into multiple dataframes and csv files.

        Parameters:
        output_dir (path): folder path where splitted data csv will be saved
        y_labels (list): labels of y we want to predict based on X
        free_whole (bool): free main data when we split into train, val, test
        train_split (float): How much data to use for training
        val_split (float): How much data to use for validation
        test_split (float): How much data to use for testing
        seed (int): seed for split, use the same for reproducibility

        Returns:
        pd.Dataframe: train, val, test
        """
        
        os.makedirs(output_dir, exist_ok=True)
        data = self.get_main_data()

        # Split into x and y for sklearn splitting
        X = data.drop(columns=y_labels)
        y = data[y_labels]

        # First, split into train (70%) and leave remaining (30%)
        # Stratify will split by Y distribution, that means distributions of y will be like in original data
        X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=(1 - train_split), stratify=y, random_state=seed)

        # Split the remaining 30% into validation and test
        # Since we set val_split and test_split based on main data, we need to recalculate split for 'remaining'
        test_size_final = (test_split / (val_split + test_split))
        X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=test_size_final, stratify=y_rem, random_state=seed)

        # Merge x and y back into one csv
        self.__train_df = pd.concat([X_train, y_train], axis=1)
        self.__val_df = pd.concat([X_val, y_val], axis=1)
        self.__test_df = pd.concat([X_test, y_test], axis=1)
        
        if self.__train_df is None or self.__val_df is None or self.__test_df is None:
            raise ValueError(f"{pemji('red_cross')} Train/Val/Test is either invalid or not set")
        
        # It is memory intensive to keep loading up memory with same data! Prefer to free!
        if free_whole is True:
            self.free_main_data()

        self.__train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        self.__val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
        self.__test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

        printc(f"{pemji('check_mark')} Train, validation, and test data splitted successfully!", 'g')
        return self.__train_df, self.__val_df, self.__test_df
        
    def load_splits(self, data_dir="splitted", train_file="train.csv", val_file="val.csv", test_file="test.csv"):
        """
        Load splitted data files into dataframes.

        Parameters:
        data_dir (path): folder path where splitted data csv's is
        train_file (string): name of train csv file inside data_dir
        val_file (string): name of val csv file inside data_dir
        test_file (string): name of test csv file inside data_dir

        Returns:
        pd.Dataframe: train, val, test
        """
        
        if not os.path.exists(data_dir):
            raise ValueError(f"{pemji('red_cross')}{pemji('')} Splitted data directory does not exist! Maybe do a split()?")
        
        self.__train_df = pd.read_csv(os.path.join(data_dir, train_file))
        if self.__data is None:
            raise ValueError(f"{pemji('red_cross')}{pemji('')} Train csv data does not exist! Maybe do a split()?")
            
        self.__val_df = pd.read_csv(os.path.join(data_dir, val_file))
        if self.__data is None:
            raise ValueError(f"{pemji('red_cross')}{pemji('')} Val csv data does not exist! Maybe do a split()?")
            
        self.__test_df = pd.read_csv(os.path.join(data_dir, test_file))
        if self.__data is None:
            raise ValueError(f"{pemji('red_cross')}{pemji('')} Test csv data does not exist! Maybe do a split()?")
        
        printc(f"{pemji('check_mark')} Train, validation, and test data loaded successfully!", 'g')
        
        return self.__train_df, self.__val_df, self.__test_df
        
    def filter_zeros(self):
        """
        Remove collumns that have no useable info for AI/ML (one collumn value).

        Parameters:
        None

        Returns:
        Nothing
        """
        
        # Know what you are filtering and working on!
        if self.data_freed is False and self.__data is not None:
            printc(f"{pemji('red_exclamation')} Filtering main data, not splitted!", 'y')
            df = self.get_main_data()
        
            original_columns = df.columns.tolist()
            original_column_count = len(original_columns)
            printc(f"{pemji('')} Original column count: {original_column_count}", 'g')
            printc(f"{pemji('')} Original columns: {original_columns}", 'v')

            # Filter out columns where the count of unique values is just 1
            df_filtered = df.loc[:, df.nunique() > 1]
            df_filtered = df_filtered.drop(columns=[''], errors='ignore') # drop unnamed cols

            self.__data = df_filtered

            filtered_columns = df_filtered.columns.tolist()
            filtered_column_count = len(filtered_columns)
            printc(f"{pemji('')} Filtered column count: {filtered_column_count}", 'g')
            printc(f"{pemji('')} Filtered columns: {filtered_columns}", 'v')

            # Determine and print the dropped column names
            dropped_columns = set(original_columns) - set(filtered_columns)
            
            printc(f"{pemji('trashcan')} Dropped columns: {dropped_columns}", 'v')
        else:
            printc(f"{pemji('red_exclamation')} Filtering splitted data, not main!", 'y')
            df_train, df_val, df_test = self.get_data()
        
            original_columns = df_train.columns.tolist()
            original_column_count = len(original_columns)
            printc(f"{pemji('')} Original column count: {original_column_count}", 'g')
            printc(f"{pemji('')} Original columns: {original_columns}", 'v')

            # Filter out columns where the count of unique values is just 1
            df_train_filtered = df_train.loc[:, df_train.nunique() > 1]
            df_val_filtered = df_val.loc[:, df_val.nunique() > 1]
            df_test_filtered = df_test.loc[:, df_test.nunique() > 1]
            
            self.__train_df = df_train_filtered
            self.__val_df = df_val_filtered
            self.__test_df = df_test_filtered

            filtered_columns = df_train_filtered.columns.tolist()
            filtered_column_count = len(filtered_columns)
            printc(f"{pemji('')} Filtered column count: {filtered_column_count}", 'g')
            printc(f"{pemji('')} Filtered columns: {filtered_columns}", 'v')

            # Determine and print the dropped column names
            dropped_columns = set(original_columns) - set(filtered_columns)
            
            printc(f"{pemji('trashcan')} Dropped columns: {dropped_columns}", 'v')
        
    def drop_collumns(self, columns):
        """
        Remove specified columns from the DataFrame.

        Parameters:
        columns (list): A list of column names to drop.

        Returns:
        Nothing
        """
        
        if self.__data is not None and self.data_freed is False:
            printc(f"{pemji('red_exclamation')} Dropping collumns on main data, not splitted!", 'y')
            columns_to_drop = [col for col in columns if col in self.__data.columns]
            self.__data = self.__data.drop(columns=columns_to_drop)
        else:
            printc(f"{pemji('red_exclamation')} Dropping collumns on splitted data, not main!", 'y')
            # Drop for each separately in case one doesn't contain some row, shouldn't happen tho
            columns_to_drop = [col for col in columns if col in self.__train_df.columns]
            self.__train_df = self.__train_df.drop(columns=columns_to_drop)
            
            columns_to_drop = [col for col in columns if col in self.__val_df.columns]
            self.__val_df = self.__val_df.drop(columns=columns_to_drop)
            
            columns_to_drop = [col for col in columns if col in self.__test_df.columns]
            self.__test_df = self.__test_df.drop(columns=columns_to_drop)
    
    def remap_columns(self, columns_to_remap=["Label",], mapping_file='mapping_info.txt'):
        """
        Remap specified columns to integer values and save mapping info to a text file.

        Parameters:
        columns_to_remap (list): A list of column strings to remap to int. Default "Label"
        mapping_file (str): The file path to save the mapping information
        """
        
        for col in columns_to_remap:
            if self.__data is not None and not self.data_freed:
                if col not in self.__data.columns:
                    printc(f"{pemji('red_cross')} '{col}' does not exist in the main data", 'r')
                    return
            else:
                if col not in self.self.__train_df.columns:
                    printc(f"{pemji('red_cross')} '{col}' does not exist in the train data", 'r')
                    return

        all_mappings = {}
        for col in columns_to_remap:
            if self.__data is not None and not self.data_freed:
                unique_labels = self.__data[col].unique()
                label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
                self.__data[col] = self.__data[col].map(label_mapping)
                all_mappings[col] = label_mapping
            else:
                for df in [self.__train_df, self.__val_df, self.__test_df]:
                    if df is not None and col in df.columns:
                        unique_labels = df[col].unique()
                        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
                        df[col] = df[col].map(label_mapping)
                        all_mappings[col] = label_mapping

        # Save the mappings so we would know later what was what
        with open(mapping_file, 'w') as f:
            for col, mapping in all_mappings.items():
                f.write(f"{col} Mapping: {mapping}\n")

        printc(f"{pemji('red_exclamation')} Label Mapping changed:\n{label_mapping}", 'y')
            
    def plot_distributions(self, plots_outputs=[(None, 'plots_main'),], max_ticks=30, filter_lower_extreme=5, filter_upper_extreme=5):
        """
        Plot distribution bar plots for data. The fast method, matplotlib sucks

        Parameters:
        plots_outputs (list of tuples): what data to use for distributions and in which folder to save distribution plots of the data
        max_ticks (int): How many bars can there be max in a distribution plot
        filter_lower_extreme (int): Percent of data at the lower extreme to remove
        filter_upper_extreme (int): Percent of data at the upper extreme to remove

        Returns:
        Nothing, but plots png's at the plots_output_dir
        """
        
        for data, out_dir in plots_outputs:
            if data is None and 'plots_main' in out_dir:
                data = self.__data
            if data is None:
                raise ValueError(f"{pemji('red_cross')}{pemji('')} Dataframe of out_dir '{out_dir}' for distribution plot is None!")

            os.makedirs(out_dir, exist_ok=True)

            for column in data.columns:
                if np.issubdtype(data[column].dtype, np.number):  # Ensure the column is numeric, else errors for distributions
                    col_data = data[column].replace([np.inf, -np.inf], np.nan).dropna()
                    
                    # This should not happen in normal scenario if data was properly filtered
                    if col_data.empty:
                        # Oopsie
                        printc(f"{pemji('exclamation_mark')} Skipping column {column}: all values are NaN, infinite, or non-numeric.", 'r')
                        continue
                    
                    # Calculate the 5% quantiles (lower and upper) to filter out extremes
                    lower_quantile = col_data.quantile(filter_lower_extreme / 100)
                    upper_quantile = col_data.quantile(1 - filter_upper_extreme / 100)
                    
                    # Get the most extreme values (before filtering, so we could print them in distrib plot)
                    lowest_value = col_data.min()
                    highest_value = col_data.max()
                    
                    # Filter the data to remove the extreme 5% on both sides
                    filtered_data = col_data[(col_data >= lower_quantile) & (col_data <= upper_quantile)]
                    
                    # This should not happen in normal scenario
                    if filtered_data.empty:
                        # Oopsie
                        printc(f"{pemji('exclamation_mark')} Skipping column {column}: all values removed after filtering extremes.", 'r')
                        continue
                    
                    # Calculate histogram bin counts and edges
                    data_ticks_here = data[column].nunique()
                    if data_ticks_here < max_ticks:
                        max_ticks_use = data_ticks_here
                    else:
                        max_ticks_use = max_ticks
                        printc(f"Max ticks reduced for '{column}' from {data_ticks_here} to: {max_ticks_use}", 'v')
                    counts, bins = np.histogram(filtered_data, bins=max_ticks_use)
                    
                    # Cap bin heights if any are above 60% of the maximum height
                    max_height = counts.max()
                    cap_threshold = 0.6 * max_height
                    
                    # Find the second tallest height and use it to cap the plot's Y size
                    second_max_height = sorted(counts)[-2] if len(counts) > 1 else max_height
                    if second_max_height == 0:
                        second_max_height = max_height # If some bin category where no values after extremes left
                        printc(f"{pemji('red_exclamation')} Unique count of {column} after filter became 1! (only vals for one category..)", 'y')
                    
                    # Set the plot height limit as 110% of the second tallest bar
                    plot_max_height = 1.1 * second_max_height
                    
                    # Do the capping and add annotations for capped collumn, how many counts of it was there
                    capped_counts = counts.copy()
                    annotations = []
                            
                    for i in range(len(capped_counts)):
                        if capped_counts[i] >= cap_threshold:
                            if capped_counts[i] == max_height:
                                capped_counts[i] = 1.0 * plot_max_height  # 100% for the tallest bar
                            else:
                                capped_counts[i] = 0.9 * plot_max_height  # 90% for the second tallest bar
                            
                            # Add annotation for original frequency
                            annotations.append(dict(
                                x=(bins[i] + bins[i + 1]) / 2,  # Position in the middle of the bin
                                y=capped_counts[i] + 0.05 * plot_max_height,  # Slightly above the bar
                                text=f"{pemji('red_exclamation')} FREQ of {bins[i]}: {counts[i]} {pemji('red_exclamation')}",
                                showarrow=True,
                                arrowhead=1,
                                ax=0,
                                ay=-40,
                            ))
                    
                    # Create the bar chart manually since we're working with counts and bins
                    fig = go.Figure()
                    # Add a bar trace using the capped counts and bin midpoints
                    bin_midpoints = (bins[:-1] + bins[1:]) / 2
                    fig.add_trace(go.Bar(x=bin_midpoints, y=capped_counts, text=capped_counts, textposition='auto'))
                    # Update layout with axis titles, ticks, and annotations
                    fig.update_layout(
                        title={
                            'text': f'Distribution of {column} (After Removing 5% Extremes)',
                            'x': 0.5,  # Center the title
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                        xaxis_title=column,
                        yaxis_title='Frequency',
                        height=720,
                        width=1280,
                        xaxis=dict(
                            tickvals=bins[:-1],
                            nticks=max_ticks_use,
                            tickangle=90, 
                            tickformat='.2s',
                        ),
                        annotations=[
                            dict(
                                xref='paper', yref='paper',
                                x=0.95, y=1.05, showarrow=False,
                                text=f"Most extreme values (before filtering):\nLowest: {lowest_value}\nHighest: {highest_value}",
                                font=dict(size=10)
                            )
                        ] + annotations
                    )
                    
                    # Add annotations for each bin
                    for annotation in annotations:
                        fig.add_annotation(annotation)

                    # Save the plot as a PNG image
                    img_file = f'{column}_distribution.png'.replace('/', '_')
                    img_path = f'{out_dir}/{img_file}'
                    fig.write_image(img_path)
                else:
                    printc(f"{pemji('exclamation_mark')} Skipping non-numeric column {column}", 'y')

            printc(f"{pemji('check_mark')} Plots saved in '{out_dir}' directory", 'g')
            
    def __save_to_excel(self, writer, df, sheet_name):
        """
        Plot distribution bar plots for data. The fast method, matplotlib sucks

        Parameters:
        writer (ExcelWriter): ExcelWriter object that contains the path to write to
        df (dataframe): dataframe data to write to excel
        sheet_name (str): name of sheet in excel to write data to

        Returns:
        Nothing, but excel sheet is saved at writer's excel file location
        """

        df.to_excel(writer, sheet_name=sheet_name, index=False)
        printc(f"{pemji('check_mark')} Statistics excel saved to sheet '{sheet_name}'", 'g')
        
    def stats(self, corr_with_collumn='Label', outputs=[(None, 'excel_main_data'),], filename='analytics.xlsx'):
        for data, output_dir in outputs:
            if data is None and "excel_main_data" in output_dir:
                data = self.__data
            os.makedirs(output_dir, exist_ok=True)
            with pd.ExcelWriter(os.path.join(output_dir, filename), engine='openpyxl') as writer:
                # Step 1: Create usual descriptive statistics
                descriptive_stats = data.describe(include='all').T
                descriptive_stats['Var'] = data.columns
                descriptive_stats['Mean'] = data.mean(numeric_only=True)  # Only numeric for fast spearman!
                descriptive_stats['Std Dev'] = data.std(numeric_only=True)
                descriptive_stats['Min'] = data.min(numeric_only=True)
                descriptive_stats['Max'] = data.max(numeric_only=True)
                descriptive_stats['Unique_alt'] = [data[col].nunique() for col in data.columns]
                descriptive_stats['Missing_values'] = [data[col].isnull().sum() for col in data.columns]
                printc(f"{pemji('check_mark')} Descriptive stats calculated", 'p')
                
                # Filter for numeric columns only, else error!
                numeric_data = data.select_dtypes(include=[np.number])

                # Check if there are enough numeric columns to compute Spearman correlation
                if numeric_data.shape[1] > 1:  # At least two numeric columns are needed for correlation
                    corr_df, p_values_df = fast_spearman(numeric_data)
                    printc(f"{pemji('check_mark')} Spearman correlations calculated", 'p')

                    # Save descriptive stats and correlation matrices to Excel
                    self.__save_to_excel(writer, descriptive_stats, 'Descriptive_Stats')
                    self.__save_to_excel(writer, corr_df.reset_index(), 'Spearman_Correlation')
                    self.__save_to_excel(writer, p_values_df.reset_index(), 'P_Values')
                    printc(f"{pemji('check_mark')} Descriptives, spearman correlations saved to Excel", 'p')
                else:
                    raise ValueError(f"{pemji('red_cross')}{pemji('')} Not enough collumns for correlations!?")

                # Step 2: Create a heatmap for Spearman correlations
                plt.figure(figsize=(40, 32))
                sns.heatmap(corr_df, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Spearman Correlation between VARs'})
                plt.title('Spearman Correlation Heatmap')
                plt.savefig('spearman_correlation_heatmap.png')
                plt.close()

                # Save the heatmap figure to Excel
                wb = writer.book
                ws = wb.create_sheet('Spearman_Heatmap')
                img = openpyxl.drawing.image.Image('spearman_correlation_heatmap.png')
                ws.add_image(img, 'A1')
                
                printc(f"{pemji('check_mark')} Heatmap for spearman drawn", 'p')

                # Step 3: Correlate each variable with the Y
                if corr_with_collumn in numeric_data.columns:  # Check if 'Label' column exists
                    label_corr_df = corr_df[[corr_with_collumn]].rename(columns={f'{corr_with_collumn}': f'Correlation with {corr_with_collumn}'})
                    
                    # Sort by absolute values to get the least correlated closer to 0
                    least_correlated = label_corr_df.reindex(label_corr_df[f'Correlation with {corr_with_collumn}'].abs().sort_values().index).head(20)

                    # For most correlated, simply sort by the actual correlation values
                    most_correlated = label_corr_df.reindex(label_corr_df[f'Correlation with {corr_with_collumn}'].abs().sort_values().index).tail(20)
                    # Reverse the list to have it from bottom to top
                    most_correlated = most_correlated.iloc[::-1]
                    
                    # Reset the index to include the column names (variables) as a new column
                    most_correlated = most_correlated.reset_index()
                    least_correlated = least_correlated.reset_index()

                    # Save these to Excel
                    self.__save_to_excel(writer, most_correlated, f'Most_Correlated_With_{corr_with_collumn}')
                    self.__save_to_excel(writer, least_correlated, f'Least_Correlated_With_{corr_with_collumn}')

                    # Step 4: Create another heatmap for correlations with the Label column
                    plt.figure(figsize=(40, 20))
                    sns.heatmap(label_corr_df, annot=True, cmap='coolwarm', cbar_kws={'label': f'Correlation with {corr_with_collumn}'})
                    plt.title(f'Correlation with {corr_with_collumn}')
                    plt.savefig(os.path.join(output_dir, f'correlation_with_{corr_with_collumn}_heatmap.png'))
                    plt.close()

                    # Save this heatmap to Excel
                    wb = writer.book
                    ws2 = wb.create_sheet(f'Correlation_With_{corr_with_collumn}_Heatmap')
                    img2 = openpyxl.drawing.image.Image(os.path.join(output_dir, f'correlation_with_{corr_with_collumn}_heatmap.png'))
                    ws2.add_image(img2, 'A1')
                    
                    # TODO: Eh, this ugly
                    if data is self.__data:
                        self.main_corr_df = corr_df
                        self.main_label_corr_df = label_corr_df
                    elif data is self.__train_df:
                        self.train_corr_df = corr_df
                        self.train_label_corr_df = label_corr_df
                    elif data is self.__val_df:
                        self.val_corr_df = corr_df
                        self.val_label_corr_df = label_corr_df
                    elif data is self.__test_df:
                        self.test_corr_df = corr_df
                        self.test_label_corr_df = label_corr_df
                    else:
                        raise ValueError(f"{pemji('red_cross')} Undefined data ptr with obj{pemji('red_exclamation')}")

                else:
                    raise ValueError(f"{pemji('red_cross')} {corr_with_collumn} does not exist to corr with{pemji('red_exclamation')}")
                    
    def remove_most_corred(self, corr_with_collumn='Label', output_dir='filtered_full', corr_threshold=0.9):
        """
        Remove most correlated collumn above threshold and keep only one that correlates more with corr_with_collumn from the pair

        Parameters:
        corr_with_collumn (str): with which collumn did we calculate correlations before! Use that here
        output_dir (path): where to write filtered csv data to
        corr_threshold (float): threshold, when to remove one collumn of data if correlation passes this threshold

        Returns:
        Nothing, but inside object the data is updated and csv files are outputed to output_dir
        """
        
        os.makedirs(output_dir, exist_ok=True)
        if self.__data is not None and not self.data_freed:
            process = [(self.__data, self.main_corr_df, self.main_label_corr_df, 'main')]
        else:
            train, val, test = self.get_data()
            process = [(train, self.main_corr_df, self.main_label_corr_df, 'train'),
                       (val, self.main_corr_df, self.main_label_corr_df, 'val'),
                       (test, self.main_corr_df, self.main_label_corr_df, 'test')]
        
        for data, corr_df, label_corr, out_type in process:
            drop_columns = set()
            for i in range(len(corr_df.columns)):
                for j in range(i + 1, len(corr_df.columns)):
                    col1, col2 = corr_df.columns[i], corr_df.columns[j]
                    if abs(corr_df.loc[col1, col2]) >= corr_threshold:
                        # Keep the variable with the higher correlation to corr_with_collumn
                        printc(f"{pemji('')} Col {col1} vs {col2} corrs at {corr_df.loc[col1, col2]}. {col1} corr with {corr_with_collumn} {label_corr.loc[col1, f'Correlation with {corr_with_collumn}']}, while {col2} corr with {corr_with_collumn} {label_corr.loc[col2, f'Correlation with {corr_with_collumn}']}", 'p')
                        if abs(label_corr.loc[col1, f'Correlation with {corr_with_collumn}']) >= abs(label_corr.loc[col2, f'Correlation with {corr_with_collumn}']):
                            printc(f"{pemji('')} Will drop {col2}", 'p')
                            drop_columns.add(col2)
                        else:
                            printc(f"{pemji('')} Will drop {col1}", 'p')
                            drop_columns.add(col1)

            printc(f"{pemji('')} Columns to be dropped due to high correlation: {drop_columns}", 'y')

            # Drop the highly correlated columns from the data
            reduced_data = data.drop(columns=drop_columns)
            self.__data = reduced_data
            
            # Save the reduced dataset to a new CSV file
            output_csv_path = os.path.join(output_dir, f'final_{out_type}.csv')
            reduced_data.to_csv(output_csv_path, index=False)
            
            printc(f"{pemji('rocket')} Final dataset saved to {output_csv_path}", 'g')
        
        printc(f"{pemji('check_mark')} Final datasets done! {pemji('check_mark')}", 'g')