import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


# Function to generate Beta distribution points using scipy
def generate_beta_distribution(alpha, beta_param, number_of_points=1, location=0, scale=1, skip_hist=True):
    """
    Generate random points from a Beta distribution using scipy.

    Parameters:
    alpha (float): Shape parameter alpha (> 0)
    beta_param (float): Shape parameter beta (> 0)
    number_of_points (int): Number of points to generate
    location (float): Location parameter (default is 0)
    scale (float): Scale parameter (default is 1)
    skip_hist (bool): Whether to skip plotting histogram or not

    Returns:
    np.array: Random points from the Beta distribution
    """
    points = beta.rvs(alpha, beta_param, loc=location, scale=scale, size=number_of_points)

    if not skip_hist:
        plot_hist(alpha, beta_param, points)

    return points


def plot_hist(alpha, beta_param, points):
    """
    Plot the histogram of generated points along with the theoretical Beta PDF.

    Parameters:
    alpha (float): Shape parameter alpha (> 0)
    beta_param (float): Shape parameter beta (> 0)
    points (np.array): Generated points from Beta distribution
    """
    # Plot the histogram of generated points
    plt.hist(points, bins=50, density=True, alpha=0.6, color='b')

    # Plot the theoretical Beta distribution (for comparison)
    x = np.linspace(min(points), max(points), 1000)
    plt.plot(x, beta.pdf(x, alpha, beta_param), 'r-', lw=2, label=f'Theoretical Beta PDF')
    plt.ylim((0, 25))
    # Add labels and title
    plt.title(f"Beta Distribution (alpha={alpha}, beta={beta_param})")
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.legend()
    plt.show()


# Parameters for Beta distribution
alpha = 0.7  # Shape parameter alpha (controls the distribution skew)
beta_param = 0.5  # Shape parameter beta (controls the distribution skew)
size = 1000  # Number of points to generate

# Generate points
points = generate_beta_distribution(alpha, beta_param, size, skip_hist=False)