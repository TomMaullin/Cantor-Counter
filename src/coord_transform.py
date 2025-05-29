import numpy as np


def spherical_to_cartesian(R, theta, psi):
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    R : array_like
        Radial distances.
    theta : array_like
        Polar angles in radians (measured from the z-axis).
    psi : array_like
        Azimuthal angles in radians (measured from the x-axis).

    Returns
    -------
    x, y, z : ndarray
        Cartesian coordinates corresponding to the input spherical coordinates.
    """
    x = R * np.sin(theta) * np.cos(psi)
    y = R * np.sin(theta) * np.sin(psi)
    z = R * np.cos(theta)
    return x, y, z
