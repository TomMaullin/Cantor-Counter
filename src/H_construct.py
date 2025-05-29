import numpy as np
from src.integral import compute_integral
from src.extend import extend_F


def spike(x, n, k):
    """
    Compute the triangular "spike" function Spike_{k,n}(x), localized to a small interval
    determined by the Cantor-like construction.

    Parameters
    ----------
    x : np.ndarray
        Input values in [0, 1].
    n : int
        Spike level (controls resolution); must be >= 0.
    k : int
        Index in {0, ..., 2^n - 1} selecting the spike position.

    Returns
    -------
    np.ndarray
        Values of the spike function at input x.
    """
    x = np.asarray(x)

    # Convert k to binary digits (r_1, ..., r_n)
    r = [(k >> (n - j - 1)) & 1 for j in range(n)]

    # Calculate interval endpoints a_{n,k} and b_{n,k}
    a_nk = sum(r_j * (3 / 5) * (2 / 5)**j for j, r_j in enumerate(r))
    b_nk = a_nk + (2 / 5)**n

    # Spike height and width
    h_n = 4 * (5 / 6)**(n + 1)
    w_n = 2 / (h_n * 3**(n + 1))

    # Spike support region
    mid = 0.5 * (a_nk + b_nk)
    left = mid - 0.5 * w_n
    right = mid + 0.5 * w_n

    # Construct spike piecewise
    y = np.zeros_like(x)

    # Rising edge
    mask_rise = (x > left) & (x <= mid)
    y[mask_rise] = 2 * h_n / w_n * (x[mask_rise] - left)

    # Falling edge
    mask_fall = (x > mid) & (x < right)
    y[mask_fall] = h_n - 2 * h_n / w_n * (x[mask_fall] - mid)

    # Optional: force exact peak height (mostly symbolic)
    y[np.isclose(x, mid)] = h_n

    return y


def compute_h(x, N_max=5):
    """
    Compute a fractal function h(x) as a finite sum of triangular spikes.

    Parameters
    ----------
    x : np.ndarray
        Array of input values in [0, 1].
    N_max : int
        Maximum spike level n (controls detail/resolution).

    Returns
    -------
    np.ndarray
        Approximated h(x) values over x.
    """
    x = np.asarray(x)
    h_vals = np.zeros_like(x)

    for n in range(N_max + 1):
        for k in range(2**n):
            h_vals += spike(x, n, k)

    return h_vals


def compute_H_tilde(x_vals, normalise=True):
    """
    Compute the integrated version of h(x), symmetrized to form \tilde{H}(x).

    Parameters
    ----------
    x_vals : np.ndarray
        Input values (may be negative or positive).
    normalise : bool, optional
        Whether to normalize so that H(1) = 1.

    Returns
    -------
    np.ndarray
        Values of \tilde{H}(x) over input x_vals.
    """
    x_vals = np.asarray(x_vals)

    # Separate into negative and non-negative values
    x_neg = x_vals[x_vals < 0]
    x_pos = x_vals[x_vals >= 0]
    H_tilde_vals = np.empty_like(x_vals, dtype=float)

    # Compute \tilde{H}(x) for negative side by symmetry
    if len(x_neg) > 0:
        x_neg_flipped = -x_neg[::-1]
        H_tilde_neg = compute_integral(x_neg_flipped, compute_h, normalise=normalise)
        H_tilde_vals[:len(H_tilde_neg)] = H_tilde_neg[::-1]

    # Compute \tilde{H}(x) for non-negative side
    if len(x_pos) > 0:
        H_tilde_pos = compute_integral(x_pos, compute_h, normalise=normalise)
        H_tilde_vals[-len(H_tilde_pos):] = H_tilde_pos

    return H_tilde_vals


def compute_H_star(x_vals, y_vals=None, normalise=True):
    """
    Compute H*(x, y) = 0.5 * (H̃(x) + H̃(y)) as a 2D function.

    Parameters
    ----------
    x_vals : np.ndarray
        1D array of x-coordinates.
    y_vals : np.ndarray, optional
        1D array of y-coordinates. Defaults to x_vals.
    normalise : bool, optional
        Whether to normalize \tilde{H}.

    Returns
    -------
    np.ndarray
        2D array representing H*(x, y) on the product grid.
    """
    if y_vals is None:
        y_vals = x_vals

    Hx = compute_H_tilde(x_vals, normalise=normalise)
    Hy = compute_H_tilde(y_vals, normalise=normalise)

    # Outer addition, then average
    H_star_vals = (Hx[:, np.newaxis] + Hy[np.newaxis, :]) / 2
    return H_star_vals


def compute_H_star_global(x_vals, y_vals):
    """
    Extend H*(x, y) smoothly beyond its original domain using a C¹ extension.

    Parameters
    ----------
    x_vals : np.ndarray
        1D array of x-coordinates.
    y_vals : np.ndarray
        1D array of y-coordinates.

    Returns
    -------
    np.ndarray
        Extended 2D array H*(x, y), globally defined on ℝ².
    """
    # Compute core H*(x, y)
    H_star_vals = compute_H_star(x_vals, y_vals)

    # Define target "outer" values for the extension
    c_vals = np.ones(H_star_vals.shape)

    # Build 2D grid for spatial coordinates
    X, Y = np.meshgrid(x_vals, y_vals, indexing='xy')

    # Extend H* smoothly outside core using helper
    H_star_vals = extend_F(H_star_vals, c_vals, X, Y, epsilon=1.0)

    return H_star_vals
