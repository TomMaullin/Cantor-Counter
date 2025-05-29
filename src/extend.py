import numpy as np


def extend_F(F_vals, c_vals, x_grid, y_grid, epsilon=1.0):
    """
    Smoothly extend a function F defined on [-1, 1]^2 to all of ℝ² using a C¹ transition.

    The extension uses an epsilon-thickened boundary around the domain:
      - Inside [-1, 1]^2:         use F_vals
      - Within epsilon-shell:     interpolate between F_vals and c_vals
      - Outside epsilon-shell:    use c_vals

    Parameters
    ----------
    F_vals : np.ndarray
        Core function values defined on [-1, 1]^2.
    c_vals : np.ndarray
        Outer (constant) values to be used outside the core region.
    x_grid, y_grid : np.ndarray
        2D coordinate grids corresponding to F_vals and c_vals.
    epsilon : float, optional
        Thickness of the transition shell between core and outer region.

    Returns
    -------
    new_vals : np.ndarray
        Smoothly extended function over the full domain.
    """
    # Transpose to align with meshgrid convention
    x_grid = x_grid.T
    y_grid = y_grid.T
    new_vals = np.zeros_like(F_vals)

    # Step 1: Compute distance to the square [-1, 1]^2
    dx = np.maximum(np.abs(x_grid) - 1, 0)
    dy = np.maximum(np.abs(y_grid) - 1, 0)
    dist = np.sqrt(dx**2 + dy**2)  # Euclidean distance to boundary

    # Step 2: Define spatial regions
    in_core = (np.abs(x_grid) <= 1) & (np.abs(y_grid) <= 1)
    in_shell = (dist <= epsilon) & (~in_core)
    outside = dist > epsilon

    # Step 3: Compute blending weights for smooth transition in the shell
    d = dist
    denom = d**2 + (1 - d)**2
    denom[denom == 0] = 1  # Prevent division by zero
    w = d**2 / denom       # Weight assigned to c_vals in transition

    # Step 4: Combine values based on region
    new_vals[in_core] = F_vals[in_core]
    new_vals[in_shell] = (
        w[in_shell] * c_vals[in_shell] +
        (1 - w[in_shell]) * F_vals[in_shell]
    )
    new_vals[outside] = c_vals[outside]

    return new_vals
