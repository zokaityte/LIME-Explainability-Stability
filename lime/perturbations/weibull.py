import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

# Function to generate Weibull distribution points using scipy
def generate_weibull_distribution(shape_param=5, scale=1, number_of_points=1, skip_hist=True):
    """
    Generate random points from a Weibull distribution using scipy.

    Parameters:
    shape_param (float): Shape parameter (c > 0)
    scale (float): Scale parameter (default 1)
    number_of_points (int): Number of points to generate
    skip_hist (bool): Whether to skip plotting histogram or not

    Returns:
    np.array: Random points from the Weibull distribution
    """
    points = weibull_min.rvs(shape_param, scale=scale, size=number_of_points)

    if not skip_hist:
        plot_hist(shape_param, scale, points)

    return points

def plot_hist(shape_param, scale, points):
    """
    Plot the histogram of generated points along with the theoretical Weibull PDF.

    Parameters:
    shape_param (float): Shape parameter (c > 0)
    scale (float): Scale parameter (default 1)
    points (np.array): Generated points from Weibull distribution
    """
    # Plot the histogram of generated points
    plt.hist(points, bins=50, density=True, alpha=0.6, color='b')

    # Plot the theoretical Weibull distribution (for comparison)
    x = np.linspace(0, max(points), 1000)
    plt.plot(x, weibull_min.pdf(x, shape_param, scale=scale), 'r-', lw=2, label=f'Theoretical Weibull PDF')

    # Add labels and title
    plt.title(f"Weibull Distribution (shape={shape_param}, scale={scale})")
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.legend()
    plt.show()

def test():
    # Parameters for Weibull distribution
    shape_param = 5  # Shape parameter
    scale = 1        # Scale parameter (default is 1)
    size = 10000       # Number of points to generate

    # Generate points
    points = generate_weibull_distribution(shape_param, scale, size, skip_hist=False)