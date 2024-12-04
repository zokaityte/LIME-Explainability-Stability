# import ast
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from matplotlib import pyplot as plt
# from lime_experiment.metrics import jaccard_similarities
#
#
# def extract_explanations(df: pd.DataFrame) -> pd.DataFrame:
#     """Processes the dataframe to extract feature lists from the 'results' column."""
#     df["results"] = df["results"].apply(lambda x: ast.literal_eval(x))
#     df["feature_list"] = df["results"].apply(lambda x: [item[0].strip() for item in x])
#     return df
#
#
# def calculate_avg_similarity_matrix(df: pd.DataFrame) -> np.ndarray:
#     """Computes the average Jaccard similarity matrix across all labels."""
#     df = extract_explanations(df)
#     all_similarity_matrices = []
#
#     for label in df.explained_label.unique():
#         label_df = df[df.explained_label == label]
#         feature_lists = label_df.feature_list.tolist()
#         similarity_matrix = jaccard_similarities(feature_lists)
#         all_similarity_matrices.append(similarity_matrix)
#
#     # Combine and compute the average similarity matrix
#     combined_matrices = np.stack(all_similarity_matrices, axis=-1)
#     avg_matrix = np.mean(combined_matrices, axis=-1)
#     return avg_matrix
#
#
# def visualize_multiple_similarity_matrices(matrices: list, titles: list):
#     """Visualizes multiple similarity matrices in a grid layout."""
#     n = len(matrices)
#     cols = 5  # Display 5 graphs in one row
#     fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 8), constrained_layout=True)
#
#     for i, (matrix, title) in enumerate(zip(matrices, titles)):
#         # Create a mask for the upper triangle
#         mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
#         sns.heatmap(
#             matrix,
#             mask=mask,
#             annot=False,
#             fmt=".2f",
#             cmap="Blues",
#             vmin=0,
#             vmax=1,
#             cbar=(i == cols // 2),  # Show colorbar only for the middle graph
#             cbar_kws={'label': "Jaccard Similarity"} if (i == cols // 2) else None,
#             ax=axes[i]
#         )
#         axes[i].set_title(title)
#         axes[i].set_xlabel("Iteration")
#         axes[i].set_ylabel("Iteration" if i == 0 else "")  # Y-axis only on the first graph
#         axes[i].tick_params(axis='y', rotation=0)
#
#     plt.suptitle("Comparison of Average Similarity Matrices", fontsize=16)
#     plt.show()
#
#
# if __name__ == "__main__":
#     # Load datasets
#     gaussian_df = pd.read_csv("explanations_gaussian_dd8a5065.csv")
#     pareto_df = pd.read_csv("explanations_pareto_f1719849.csv")
#     gamma_df = pd.read_csv("explanations_gamma.csv")
#     beta_df = pd.read_csv("explanations_beta.csv")
#     weibull_df = pd.read_csv("explanations_weibull.csv")
#
#     # Calculate similarity matrices
#     gaussian_avg_matrix = calculate_avg_similarity_matrix(gaussian_df)
#     pareto_avg_matrix = calculate_avg_similarity_matrix(pareto_df)
#     gamma_avg_matrix = calculate_avg_similarity_matrix(gamma_df)
#     beta_avg_matrix = calculate_avg_similarity_matrix(beta_df)
#     weibull_avg_matrix = calculate_avg_similarity_matrix(weibull_df)
#
#     # Visualize all matrices
#     visualize_multiple_similarity_matrices(
#         [gaussian_avg_matrix, pareto_avg_matrix, gamma_avg_matrix, beta_avg_matrix, weibull_avg_matrix],
#         ["Gaussian", "Pareto", "Gamma", "Beta", "Weibull"]
#     )
import os

from rootutils import rootutils

rootutils.setup_root(__file__, indicator=['pyproject.toml'], pythonpath=True)

import ast
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from lime_experiment.metrics import jaccard_similarities


def extract_explanations(df: pd.DataFrame) -> pd.DataFrame:
    """Processes the dataframe to extract feature lists and explanation scores from the 'results' column."""
    # Check and parse the 'results' column only if necessary
    df["results"] = df["results"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    # Extract feature names for similarity matrix computation
    df["feature_list"] = df["results"].apply(lambda x: [item[0].strip() for item in x])
    # Extract feature names and scores for plotting
    df["feature_scores"] = df["results"].apply(lambda x: [(item[0].strip(), item[1]) for item in x])
    return df



def calculate_avg_similarity_matrix(df: pd.DataFrame) -> np.ndarray:
    """Computes the average Jaccard similarity matrix across all labels."""
    df = extract_explanations(df)
    all_similarity_matrices = []

    for label in df.explained_label.unique():
        label_df = df[df.explained_label == label]
        feature_lists = label_df.feature_list.tolist()
        similarity_matrix = jaccard_similarities(feature_lists)

        #print first and second iteration and its pairwise score
        print("Label: ",label, "first iteration", label_df.results.iloc[0])
        print("Label: ", label, "second iteration", label_df.results.iloc[1])
        print("Jaccard Similarity: ", similarity_matrix[0, 1])

        all_similarity_matrices.append(similarity_matrix)

    # Combine and compute the average similarity matrix
    combined_matrices = np.stack(all_similarity_matrices, axis=-1)
    avg_matrix = np.mean(combined_matrices, axis=-1)
    return avg_matrix


def plot_explanation_features(feature_scores: list, title: str, ax, highlight_labels: list = None, bg_color="lightblue"):
    """
    Plots explanation features as a bar chart with a colored background for highlighted tick labels.

    Args:
        feature_scores (list): List of tuples (Feature, Score).
        title (str): Title of the chart.
        ax: Matplotlib axis object.
        highlight_labels (list, optional): List of feature labels to highlight. Defaults to None.
        bg_color (str, optional): Background color for highlighted tick labels. Defaults to "lightblue".
    """
    # Convert the list of tuples into a DataFrame
    df_features = pd.DataFrame(feature_scores, columns=['Feature', 'Score'])
    # Reverse the order of the features
    df_features = df_features.iloc[::-1]

    # Assign default colors (green for positive, red for negative)
    colors = df_features['Score'].apply(lambda x: 'green' if x > 0 else 'red')

    # Plot the horizontal bar chart
    ax.barh(
        y=df_features['Feature'],
        width=df_features['Score'],
        color=colors,
        height=0.9  # Reduce bar spacing by setting height closer to 1
    )

    # Highlight labels on the y-axis with background colors
    for tick_label in ax.get_yticklabels():
        label_text = tick_label.get_text()
        if highlight_labels and label_text in highlight_labels:
            tick_label.set_color('black')  # Ensure text is visible
            tick_label.set_fontweight('bold')  # Bold text
            tick_label.set_bbox(dict(facecolor=bg_color, edgecolor='none', boxstyle='round,pad=0.3'))

    # Configure chart details
    ax.set_title(title)



# Optionally remove gridlines for a cleaner look



def visualize_summary(df: pd.DataFrame, avg_matrix: np.ndarray, name: str):
    """Creates a 3-row plot: iteration 1 features, iteration 2 features, average similarity matrix."""
    # Ensure explanations are extracted
    df = extract_explanations(df)

    # Extract features and scores for the first and second iterations
    iteration1_row = df[(df["test_run_number"] == 1) & (df["explained_label"] == 'explained_label_0')]
    iteration2_row = df[(df["test_run_number"] == 2) & (df["explained_label"] == 'explained_label_0')]

    if iteration1_row.empty or iteration2_row.empty:
        print("Data for the specified iterations and explained_label is not available.")
        return

    iteration1_features_scores = iteration1_row["feature_scores"].iloc[0]
    iteration2_features_scores = iteration2_row["feature_scores"].iloc[0]

    # Extract feature lists from each iteration
    iteration1_features = set(iteration1_row["feature_list"].iloc[0])
    iteration2_features = set(iteration2_row["feature_list"].iloc[0])

    # Find features unique to each iteration
    unique_to_iteration1 = iteration1_features - iteration2_features
    unique_to_iteration2 = iteration2_features - iteration1_features

    # Convert to lists for highlighting
    unique_to_iteration1_list = list(unique_to_iteration1)
    unique_to_iteration2_list = list(unique_to_iteration2)

    # Create the figure with a 3-row grid, adjusting heights and widths
    from matplotlib import gridspec

    fig = plt.figure(figsize=(10, 15))
    gs = gridspec.GridSpec(3, 2, width_ratios=[0.05, 0.95], height_ratios=[0.75, 0.75, 2], wspace=0.25, hspace=0.25)

    # Axes for the bar charts
    axes_iteration1 = fig.add_subplot(gs[0, 1])
    axes_iteration2 = fig.add_subplot(gs[1, 1])

    # Axes for heatmap and its color bar
    axes_heatmap = fig.add_subplot(gs[2, 1])
    cbar_ax = fig.add_subplot(gs[2, 0])

    # Plot first iteration features with highlights
    plot_explanation_features(
        iteration1_features_scores,
        "Local explanation for class Benign (run=0)",
        axes_iteration1,
        highlight_labels=unique_to_iteration1_list
    )

    # Plot second iteration features with highlights
    plot_explanation_features(
        iteration2_features_scores,
        "Local explanation for class Benign (run=1)",
        axes_iteration2,
        highlight_labels=unique_to_iteration2_list
    )

    # Plot average similarity matrix
    mask = np.triu(np.ones_like(avg_matrix, dtype=bool), k=1)
    sns.heatmap(
        avg_matrix,
        mask=mask,
        annot=False,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar=True,
        ax=axes_heatmap,
        cbar_ax=cbar_ax,  # Specify the color bar axis
        cbar_kws={'label': "Jaccard Similarity"}
    )
    axes_heatmap.set_title("Average Similarity Matrix")
    cbar_ax.yaxis.set_label_position("left")
    cbar_ax.yaxis.tick_left()

    plt.suptitle(name, fontsize=16)
    plt.show()
    plt.savefig(name + ".png")





if __name__ == "__main__":
    # List of CSV file names
    csv_files = [
        "exp_DecisionTreeClassifier_gaussian_dd8a5065.csv",
        "exp_DecisionTreeClassifier_beta_f809c4d6.csv",
        "exp_DecisionTreeClassifier_gamma_7bc3721c.csv",
        "exp_DecisionTreeClassifier_pareto_8a232a1c.csv",
        "exp_DecisionTreeClassifier_weibull_eb69266e.csv"
    ]

    print(os.getcwd())

    for csv_file in csv_files:
        # Load the dataset
        df = pd.read_csv(csv_file)

        # Calculate the average similarity matrix
        avg_matrix = calculate_avg_similarity_matrix(df)

        # Visualize summary
        visualize_summary(df, avg_matrix, csv_file)
