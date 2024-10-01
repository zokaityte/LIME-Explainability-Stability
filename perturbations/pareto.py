import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pareto

# Function to generate Pareto distribution points using scipy
def generate_pareto_distribution(shape_param, scale=1, number_of_points=1, skip_hist=True):
    """
    Generate random points from a Pareto distribution using scipy.

    Parameters:
    shape_param (float): Shape parameter a (> 0)
    scale (float): Scale parameter (default 1)
    number_of_points (int): Number of points to generate
    skip_hist (bool): Whether to skip plotting histogram or not

    Returns:
    np.array: Random points from the Pareto distribution
    """
    points = pareto.rvs(shape_param, scale=scale, size=number_of_points)

    if not skip_hist:
        plot_hist(shape_param, scale, points)

    return points

def plot_hist(shape_param, scale, points):
    """
    Plot the histogram of generated points along with the theoretical Pareto PDF.

    Parameters:
    shape_param (float): Shape parameter a (> 0)
    scale (float): Scale parameter (default 1)
    points (np.array): Generated points from Pareto distribution
    """
    # Plot the histogram of generated points
    plt.hist(points, bins=50, density=True, alpha=0.6, color='b')

    # Plot the theoretical Pareto distribution (for comparison)
    x = np.linspace(min(points), max(points), 1000)
    plt.plot(x, pareto.pdf(x, shape_param, scale=scale), 'r-', lw=2, label=f'Theoretical Pareto PDF')

    plt.ylim((0, 1.25))
    # Add labels and title
    plt.title(f"Pareto Distribution (shape={shape_param}, scale={scale})")
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.legend()
    plt.show()


# Parameters for Pareto distribution
shape_param = 2.5  # Shape parameter (also called "alpha" or "a")
scale = 1.0        # Scale parameter (default is 1)
size = 10000       # Number of points to generate

# Generate points
points = generate_pareto_distribution(shape_param, scale, size, skip_hist=False)
