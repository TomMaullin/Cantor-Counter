import numpy as np
import plotly.graph_objects as go
from src.f_construct import compute_f_star
from src.coord_transform import spherical_to_cartesian

def plot_surface_3d(F_vals, x_vals, y_vals=None, downsample=1, title='Surface Plot', z_title='Z', sphere_radius=None):
    """
    Plots a 3D surface for F_vals over the given x and y values.

    Parameters:
    - F_vals: 2D array of values representing F(x, y)
    - x_vals: 1D array for x-axis (or 2d precomputed grid coordinates)
    - y_vals: 1D array for y-axis (or 2d precomputed grid coordinates) (optional; defaults to x_vals)
    - downsample: int factor to reduce resolution (e.g., 2 means every 2nd point)
    - sphere_radius: float, optional — if provided, a translucent sphere of this radius is added
    """

    if y_vals is None:
        y_vals = x_vals

    if y_vals.ndim != x_vals.ndim:
        raise ValueError('x and y must have same number of dimensions.')

    # Downsample data if needed
    if downsample > 1:
        F_vals = F_vals[::downsample, ::downsample]

        if x_vals.ndim == 1:
            x_vals = x_vals[::downsample]
            y_vals = y_vals[::downsample]
        else:
            x_vals = x_vals[::downsample, ::downsample]
            y_vals = y_vals[::downsample, ::downsample]

    # Create 2D grid (X, Y)
    if x_vals.ndim == 1:
        X, Y = np.meshgrid(x_vals, y_vals, indexing='xy')
    else:
        X = x_vals
        Y = y_vals

    # Create surface plot list
    surfaces = [
        go.Surface(
            x=X,
            y=Y,
            z=F_vals,
            colorscale='Viridis',
            name='Main Surface'
        )
    ]

    # Optionally add a translucent sphere
    if sphere_radius is not None:
        u = np.linspace(0, np.pi, 50)
        v = np.linspace(0, 2 * np.pi, 50)
        U, V = np.meshgrid(u, v)
        Xs = sphere_radius * np.sin(U) * np.cos(V)
        Ys = sphere_radius * np.sin(U) * np.sin(V)
        Zs = sphere_radius * np.cos(U)

        surfaces.append(
            go.Surface(
                x=Xs,
                y=Ys,
                z=Zs,
                opacity=0.3,
                showscale=False,
                colorscale='Blues',
                name=f'Sphere r={sphere_radius}'
            )
        )

    # Create figure
    fig = go.Figure(data=surfaces)

    # Layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title=z_title
        ),
        autosize=True,
        width=800,
        height=700
    )

    fig.show()


def plot_slice_of_f_star(
    axis='z',
    slice_value=0.0,
    x_range=np.linspace(-4, 4, 500),
    y_range=np.linspace(-4, 4, 500),
    z_range=np.linspace(0, 1.5, 500)
):
    assert axis in {'x', 'y', 'z'}, "axis must be 'x', 'y', or 'z'"

    if axis == 'z':
        f_vals = compute_f_star(x_range, y_range, np.array([slice_value]))
        f_slice = f_vals[:, :, 0]
        X, Y = np.meshgrid(x_range, y_range, indexing='ij')
        xlabel, ylabel = 'x', 'y'

    elif axis == 'y':
        f_vals = compute_f_star(x_range, np.array([slice_value]), z_range)
        f_slice = f_vals[:, 0, :]
        X, Y = np.meshgrid(x_range, z_range, indexing='ij')
        xlabel, ylabel = 'x', 'z'

    elif axis == 'x':
        f_vals = compute_f_star(np.array([slice_value]), y_range, z_range)
        f_slice = f_vals[0, :, :]
        X, Y = np.meshgrid(y_range, z_range, indexing='ij')
        xlabel, ylabel = 'y', 'z'
    
    # Ensure all arrays are float64 and same shape
    X = X.astype(float)
    Y = Y.astype(float)
    Z = f_slice.astype(float)

    if not (X.shape == Y.shape == Z.shape):
        raise ValueError(f"Shape mismatch: X{X.shape}, Y{Y.shape}, Z{Z.shape}")

    # Replace any NaNs or infs just in case
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    # Plot
    fig = go.Figure(data=[
        go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')
    ])

    fig.update_layout(
        title=f"Slice of f*(x, y, z) at {axis} = {slice_value}",
        scene=dict(
            xaxis=dict(title=xlabel),
            yaxis=dict(title=ylabel),
            zaxis=dict(title='f*', range=[-3, 1]),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5) 
            )
        ),
        autosize=True,
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    
    fig.add_trace(go.Surface(
    x=X, y=Y, z=Z,
    colorscale='Viridis',
    cmin=-3,  # Set this to raise the lower bound
    cmax=1,  # Optional: set upper bound to stretch contrast
    showscale=False,
    contours=dict(
        z=dict(show=True, start=0, end=0.01, size=1e-1, color='red', width=6)
    ),
    opacity=0.95
    ))
    
    fig.show()


def plot_z_slice_of_f(c,res=200):

    
    # Initial R theta and psi
    R = np.linspace(0,0.6,res)
    theta = np.linspace(0,np.pi,res)
    psi = np.linspace(0,2*np.pi,res)
    
    # Scaling factor
    k=4
    
    # Conversion for computing f*
    R_converted = np.array(R)
    theta_converted = k*(2*theta/np.pi-1)
    psi_converted = k*(psi/np.pi -1)
    
    # Compute f_vals
    f_vals = compute_f_star(theta_converted, psi_converted, R_converted)

    # COmpute meshgrid
    R_grid, theta_grid, psi_grid = np.meshgrid(R, theta, psi, indexing='ij')
    
    # Compute Cartesian coordinates from meshgrid
    x, y, z = spherical_to_cartesian(R_grid, theta_grid, psi_grid)

    # Divide c by 3
    c = c/3
    
    z_min = np.nanmin(f_vals)
    z_max = np.nanmax(f_vals)
    
    
    # Transpose to match (R, theta, psi)
    f_vals_interp = np.transpose(f_vals, (2, 0, 1)) 

    # Useful scipy function for interpolation
    from scipy.interpolate import RegularGridInterpolator
    interp_spherical = RegularGridInterpolator(
        (R, theta, psi),
        f_vals_interp,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    
    # Define grid in x and y
    grid_x, grid_y = np.mgrid[
        x.min():x.max():res*1j,
        y.min():y.max():res*1j
    ]
    grid_z = np.full_like(grid_x, c)

    # Work out points to evaluate in spherical coordinates
    R_eval = np.sqrt(grid_x**2 + grid_y**2 + grid_z**2)
    theta_eval = np.arccos(np.divide(grid_z, R_eval, out=np.zeros_like(grid_z), where=R_eval!=0))
    psi_eval = np.arctan2(grid_y, grid_x) % (2 * np.pi)
    
    # Interpolate
    points = np.stack([R_eval, theta_eval, psi_eval], axis=-1)
    grid_f = interp_spherical(points)
    
    # Mask out values where R > 0.1
    mask = R_eval > 1/3
    grid_f_masked = np.where(mask, np.nan, grid_f)
    
    fig = go.Figure(data=go.Surface(
        x=3*grid_x, # Rescaled to match final function
        y=3*grid_y, # Rescaled to match final function
        z=grid_f_masked,  # This is the function value f(x, y, z=c)
        colorscale='Viridis',
        cmin=-0.15,  # Set this to raise the lower bound
        cmax=0.25,  # Optional: set upper bound to stretch contrast
        showscale=True,
        contours=dict(
            z=dict(show=True, start=0, end=0.01, size=1e-1, color='red', width=6)
        ),
        colorbar=dict(title='f(x, y, z)')
    ))
    
    fig.update_layout(
        title=f'Function Slice at z = {3*c}', # Rescaled by 3 to match final function
        scene=dict(
            xaxis=dict(title='x', range=[-1, 1]),
            yaxis=dict(title='y', range=[-1, 1]),
            zaxis=dict(title='f(x, y, z)', range=[z_min-0.03, z_max]),
            camera=dict(
                eye=dict(x=-2, y=1, z=1) 
            )
        ),
        width=800,
        height=600
    )

    fig.show()
