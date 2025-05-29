import numpy as np
from src.T_construct import compute_T_star
from src.H_construct import compute_H_star_global


def compute_f_star(x_values, y_values, z_values):
    """
    Compute the function f*(x, y, z) as a smooth embedding of the surface 
    z = H*(x, y) into ℝ³, with specific control over the gradient.

    The function is defined piecewise with two smooth terms:
      f*(x, y, z) = -T*(x, y) * (z - H*(x, y)) * z²
                    - 0.25 * (z - H*(x, y))²  if z > H*
                    0                          otherwise

    Parameters
    ----------
    x_values : array_like
        1D array of x-coordinates.
    y_values : array_like
        1D array of y-coordinates.
    z_values : array_like
        1D array of z-coordinates.

    Returns
    -------
    f_star : ndarray
        3D array of shape (Nx, Ny, Nz) representing f*(x, y, z).
    """
    # Convert inputs to NumPy arrays
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    z_values = np.asarray(z_values)

    # Compute T*(x, y), shape (Nx, Ny), then expand for broadcasting to (Nx, Ny, 1)
    T_sum = compute_T_star(x_values, y_values)[:, :, np.newaxis]

    # Compute H*(x, y), shape (Nx, Ny), then expand to (Nx, Ny, 1)
    H_star = compute_H_star_global(x_values, y_values)[:, :, np.newaxis]

    # Expand z_values to broadcast with x and y: shape (1, 1, Nz)
    Z = z_values[np.newaxis, np.newaxis, :]

    # Compute Z - H*(x, y), broadcast to shape (Nx, Ny, Nz)
    z_minus_H = Z - H_star

    # Compute Z²: shape (1, 1, Nz)
    Z2 = Z * Z

    # Compute the first term: -T_sum * (Z - H*) * Z²
    term1 = T_sum * z_minus_H
    del T_sum  # Free memory
    term1 *= Z2  # In-place multiplication to reduce memory allocation

    # Compute second term: -0.25 * (Z - H*)², applied only where z > H*
    z_minus_H_sq = (z_minus_H * z_minus_H) / 4
    z_minus_H_sq[Z <= H_star] = 0  # Mask out values where z ≤ H*

    # Final result: f* = -2 * term1 - z_minus_H_sq
    f_star = -2 * term1 - z_minus_H_sq

    # Clean up intermediates to save memory
    del term1, z_minus_H_sq

    return f_star  # Shape: (Nx, Ny, Nz)
