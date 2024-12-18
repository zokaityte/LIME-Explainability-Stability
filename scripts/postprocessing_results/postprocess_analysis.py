import ast

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from lime_experiment.metrics import jaccard_similarities

plt.style.use("ggplot")

def extract_explanations(df: pd.DataFrame) -> pd.DataFrame:
    """Processes the dataframe to extract feature lists and explanation scores from the 'results' column."""
    df["results"] = df["results"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df["feature_list"] = df["results"].apply(lambda x: [item[0].strip() for item in x])
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
        all_similarity_matrices.append(similarity_matrix)

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
    df_features = pd.DataFrame(feature_scores, columns=['Feature', 'Score'])
    df_features = df_features.iloc[::-1]

    # Extract ggplot colors
    ggplot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    green_color = ggplot_colors[5]  # ggplot green
    red_color = ggplot_colors[0]  # ggplot red

    # Assign colors based on the score
    colors = df_features['Score'].apply(lambda x: green_color if x > 0 else red_color)

    # Plot the horizontal bar chart
    ax.barh(
        y=df_features['Feature'],
        width=df_features['Score'],
        color=colors,
        height=0.9# Reduce bar spacing by setting height closer to 1
    )

    # Formatting
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(ax.xaxis.label.get_color())  # Set box color to black

    # Highlight labels on the y-axis with background colors
    for tick_label in ax.get_yticklabels():
        label_text = tick_label.get_text()
        if highlight_labels and label_text in highlight_labels:
            tick_label.set_color('black')  # Ensure text is visible
            tick_label.set_fontweight('bold')  # Bold text
            tick_label.set_bbox(dict(facecolor=bg_color, edgecolor='none', boxstyle='round,pad=0.3'))

    ax.set_title(title)


def visualize_summary(df: pd.DataFrame, avg_matrix: np.ndarray, name: str):
    """Creates a layout with iteration bar charts on the left and similarity matrix on the right."""
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

    # Create the figure with a custom layout
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[0.5, 0.5])

    # Axes for bar charts (left side)
    axes_iteration1 = fig.add_subplot(gs[0, 0])  # Top-left
    axes_iteration2 = fig.add_subplot(gs[1, 0])  # Bottom-left

    # Axis for similarity matrix (right side)
    axes_heatmap = fig.add_subplot(gs[:, 1])  # Spanning both rows on the right

    # Plot first iteration features with highlights
    plot_explanation_features(
        iteration1_features_scores,
        "Local explanation \n for class Benign (run=0)",
        axes_iteration1,
        highlight_labels=unique_to_iteration1_list
    )

    # Plot second iteration features with highlights
    plot_explanation_features(
        iteration2_features_scores,
        "Local explanation \n for class Benign (run=1)",
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
        cbar_kws={'label': "Jaccard Similarity"}
    )
    axes_heatmap.set_title("Average Similarity Matrix")
    axes_heatmap.set_facecolor("white")

    # Adjust layout
    plt.suptitle(name, fontsize=16)
    plt.tight_layout()
    plt.savefig(name + ".png")
    plt.show()


if __name__ == "__main__":
    csv_files = [
        "exp_DecisionTreeClassifier_gaussian_dd8a5065.csv",
        "exp_DecisionTreeClassifier_beta_f809c4d6.csv",
        "exp_DecisionTreeClassifier_gamma_7bc3721c.csv",
        "exp_DecisionTreeClassifier_pareto_8a232a1c.csv",
        "exp_DecisionTreeClassifier_weibull_eb69266e.csv"
    ]

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        avg_matrix = calculate_avg_similarity_matrix(df)
        visualize_summary(df, avg_matrix, csv_file)
