import numpy as np
from typing import List, Set


def _jaccard_similarity(set1: Set, set2: Set) -> float:
    """
    Computes the Jaccard similarity between two sets.

    Args:
    set1 (set): First set of features.
    set2 (set): Second set of features.

    Returns:
    float: Jaccard similarity value between the two sets (0 to 1).
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0
    return intersection / union


def jaccard_similarities(list_of_lists_of_features: List[List]) -> np.ndarray:
    """
    Computes the Jaccard similarity matrix for a list of feature sets.

    Args:
    list_of_lists_of_features (list of lists): A list where each element is a list of features.

    Returns:
    numpy.ndarray: A symmetric matrix where element [i, j] is the Jaccard similarity
                   between feature set i and feature set j.
    """
    sets_of_features = [set(features) for features in list_of_lists_of_features]
    n = len(sets_of_features)

    # Pre-allocate a similarity matrix
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            sim = _jaccard_similarity(sets_of_features[i], sets_of_features[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim  # Symmetric matrix, fill both [i, j] and [j, i]

    return sim_matrix


def calculate_stability(list_of_lists_of_features: List[List]) -> float:
    """
    Calculates the mean of distinct pairwise Jaccard similarities (excluding self-similarities).

    Args:
    list_of_lists_of_features (list of lists): A list where each element is a list of features.

    Returns:
    float: The mean Jaccard similarity of all distinct pairwise comparisons.
    """
    sim_matrix = jaccard_similarities(list_of_lists_of_features)
    # Calculate the mean of all similarity scores in the upper triangle
    return np.mean(sim_matrix[np.triu_indices(len(sim_matrix), 1)])
