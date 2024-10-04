import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Function to generate Gamma distribution points using scipy
def generate_gamma_distribution(shape_param, scale=1, number_of_points=1, skip_hist=True):
    """
    Generate random points from a Gamma distribution using scipy.

    Parameters:
    shape_param (float): Shape parameter k (> 0)
    scale (float): Scale parameter theta (> 0, default 1)
    number_of_points (int): Number of points to generate
    skip_hist (bool): Whether to skip plotting histogram or not

    Returns:
    np.array: Random points from the Gamma distribution
    """
    points = gamma.rvs(shape_param, scale=scale, size=number_of_points)

    if not skip_hist:
        plot_hist(shape_param, scale, points)

    return points

def plot_hist(shape_param, scale, points):
    """
    Plot the histogram of generated points along with the theoretical Gamma PDF.

    Parameters:
    shape_param (float): Shape parameter k (> 0)
    scale (float): Scale parameter theta (> 0)
    points (np.array): Generated points from Gamma distribution
    """
    # Plot the histogram of generated points
    plt.hist(points, bins=50, density=True, alpha=0.6, color='b')

    # Plot the theoretical Gamma distribution (for comparison)
    x = np.linspace(min(points), max(points), 1000)
    plt.plot(x, gamma.pdf(x, shape_param, scale=scale), 'r-', lw=2, label=f'Theoretical Gamma PDF')

    # Add labels and title
    plt.title(f"Gamma Distribution (shape={shape_param}, scale={scale})")
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.legend()
    plt.show()


# Parameters for Gamma distribution
shape_param = 7.0  # Shape parameter k (also called "alpha")
scale = 2.0        # Scale parameter theta (default is 1)
size = 10000       # Number of points to generate

# Generate points
points = generate_gamma_distribution(shape_param, scale, size, skip_hist=False)