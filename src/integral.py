import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d


def compute_integral(x_vals, f_function, resolution=500, grid_threshold=50, normalise=True):
    """
    Compute the integral F(x) = ∫₀ˣ f(t) dt for each x in x_vals.

    The function uses two strategies depending on the density of x_vals:
    - If x_vals is already a sufficiently dense grid, integrate directly.
    - Otherwise, compute the integral on a fine internal grid and interpolate.

    Parameters
    ----------
    x_vals : array_like
        Input array of non-negative x values at which to compute the integral.
    f_function : callable
        Function handle accepting a 1D array and returning f(x).
    resolution : int, optional
        Number of points in the internal grid (used if x_vals is sparse).
    grid_threshold : int, optional
        Number of points above which x_vals is assumed to be a dense grid.
    normalise : bool, optional
        Whether to normalize the integral so that F(1) = 1. Useful for avoiding
        cumulative numerical errors.

    Returns
    -------
    F_vals : ndarray
        Array of evaluated integral values F(x) at each x in x_vals.
    """
    x_vals = np.asarray(x_vals)
    assert np.all(x_vals >= 0), "x_vals must be non-negative"

    # Case 1: x_vals is already a sufficiently dense grid on [0, 1]
    if len(x_vals) > grid_threshold and np.sum(x_vals[x_vals > 0] < 1) > 20:
        f_vals = f_function(x_vals)
        F_vals = cumulative_trapezoid(f_vals, x_vals, initial=0)

        if normalise:
            F_vals = F_vals / F_vals[-1]

        return F_vals

    # Case 2: sparse input — use internal grid for accuracy
    x_grid = np.linspace(0, 1, resolution)
    f_grid = f_function(x_grid)

    F_grid = cumulative_trapezoid(f_grid, x_grid, initial=0)

    if normalise:
        F_grid = F_grid / F_grid[-1]

    # Interpolate the cumulative integral onto the input x_vals
    interp_func = interp1d(
        x_grid,
        F_grid,
        kind='nearest',
        bounds_error=False,
        fill_value="extrapolate"
    )

    F_vals = interp_func(x_vals)
    return F_vals
