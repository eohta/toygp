import numpy as np

def generate_grid(num_x0_pts, num_x1_pts, x0_min=-5, x0_max=5, x1_min=-5, x1_max=5):

    x0 = np.linspace(x0_min, x0_max, num_x0_pts)
    x1 = np.linspace(x1_min, x1_max, num_x1_pts)

    X0_grid, X1_grid = np.meshgrid(x0, x1, indexing='ij')

    X = np.vstack((X0_grid.flatten(), X1_grid.flatten())).T

    return X, x0, x1
