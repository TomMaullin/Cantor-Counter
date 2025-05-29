import numpy as np
from src.extend import extend_F


def compute_T_tilde(x_values):
    """
    Compute the even extension of the transition function T: ℝ → [0, 1].

    The function is defined as:
      - T(x) = 0                      if |x| == 0
      - T(x) = x² / [x² + (1 - x)²]   if 0 < |x| < 1
      - T(x) = 1                      if |x| ≥ 1

    Parameters
    ----------
    x_values : array_like
        Input array of real values.

    Returns
    -------
    T_values : ndarray
        Evaluated values of the extended transition function.
    """
    x_abs = np.abs(x_values)
    T_values = np.zeros_like(x_abs)

    # Case 1: x_abs == 0
    T_values[x_abs == 0] = 0

    # Case 2: 0 < x_abs < 1
    mask_middle = (x_abs > 0) & (x_abs < 1)
    x_mid = x_abs[mask_middle]
    T_values[mask_middle] = x_mid**2 / (x_mid**2 + (1 - x_mid)**2)

    # Case 3: x_abs >= 1
    T_values[x_abs >= 1] = 1

    return T_values


def compute_T_star(x_vals, y_vals):
    """
    Construct the 2D transition function T*: ℝ² → [0, 1] by symmetrizing T̃.

    This function combines T̃(x) and T̃(y), then applies a C¹ extension
    to ensure smoothness outside the core region.

    Parameters
    ----------
    x_vals : array_like
        1D array of x-values.
    y_vals : array_like
        1D array of y-values.

    Returns
    -------
    T_vals : ndarray
        2D array of extended T*(x, y) values.
    """
    # Compute T̃(x) and T̃(y)
    Tx = compute_T_tilde(x_vals)
    Ty = compute_T_tilde(y_vals)

    # Reshape for broadcasting
    Tx = Tx.reshape(len(x_vals), 1)
    Ty = Ty.reshape(1, len(y_vals))

    # Compute symmetrized average
    T_vals = (Tx + Ty) / 2

    # Construct 2D coordinate grids
    X, Y = np.meshgrid(x_vals, y_vals, indexing='xy')

    # Constant array for extension target
    c_vals = np.ones_like(T_vals)

    # Smoothly extend T* to the outer region
    T_vals = extend_F(T_vals, c_vals, X, Y, epsilon=1.0)

    return T_vals
