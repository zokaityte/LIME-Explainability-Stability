import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy, levy_l, levy_stable, cauchy


def generate_alpha_stable_points(alpha, beta, location=0, scale=1, number_of_points=1, skip_hist=True):
    """
    Generate random points from levy distribution.
    stability alpha = 1/2
    skewness beta = 1

    Parameters:
    alpha (float): Stability parameter (0 < alpha <= 2)
    beta (float): Skewness parameter (-1 <= beta <= 1)
    location (float)
    scale (float): Skewness parameter (sigma > 0)
    number_of_points (int): Number of points to generate

    Returns:
    np.array: Random points from the alpha stable distribution
    """
    points = None

    if alpha == 0.5 and beta == -1:
        points = levy_l.rvs(location, scale, number_of_points)
    elif alpha == 0.5 and beta == 1:
        points = levy.rvs(location, scale, number_of_points)
    elif alpha == 1 and beta == 0:
        points = cauchy.rvs(location, scale, number_of_points)
    else:
        points = levy_stable.rvs(alpha, beta, location, scale, number_of_points)

    if not skip_hist:
        plot_hist(alpha, beta, points)

    return points

def plot_hist(alpha, beta, points):
    # Plot the histogram of generated points
    plt.hist(points, bins=50, density=True, alpha=0.6, color='b')

    # Plot the theoretical Beta distribution (for comparison)
    x = np.linspace(min(points), max(points), 1000)
    # Plot the points
    plt.plot(x, levy_stable.pdf(x, alpha, beta), 'r-', lw=2, label=f'Theoretical Alpha stable PDF')

    # Add labels and title
    plt.title(f"Beta Distribution (alpha={alpha}, beta={beta})")
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.legend()
    plt.show()

alpha = 1.5  # Stability parameter (e.g., 0.5 for Levy distribution)
beta = 0.0   # Symmetry (0 for symmetric distribution)
size = 100  # Number of points to generate

# Generate points
points = generate_alpha_stable_points(alpha, beta,0, 1, size, False)