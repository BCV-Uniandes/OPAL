import numpy as np
from scipy.stats import wasserstein_distance


def get_xrd(x):
    scatter = x.Scatter
    scatter._scattering_min_twotheta = 5
    scatter._scattering_max_twotheta = 90
    two_theta, intensities, _ = scatter.powder(units="twotheta")
    area = np.trapz(intensities, two_theta)
    if area > 0:
        intensities = intensities / area
    return two_theta, intensities


def xrd_metric(a, b):
    grid_a, xrd_a = get_xrd(a)
    grid_b, xrd_b = get_xrd(b)
    return wasserstein_distance(grid_a, grid_b, u_weights=xrd_a, v_weights=xrd_b)
