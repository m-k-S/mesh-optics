#!/usr/bin/env python3
"""
3D visualization utilities using PyVista for optical ray tracing.
Handles mesh rendering, ray path visualization, and intensity mapping.
"""

import numpy as np
import pyvista as pv
from typing import List, Optional, Tuple

def tris_to_polydata(tris: np.ndarray) -> pv.PolyData:
    """
    Convert triangle array to PyVista PolyData with vertex deduplication.
    
    Parameters:
    - tris: (N, 3, 3) array of triangle vertices
    
    Returns PyVista PolyData mesh.
    """
    # Flatten and deduplicate vertices
    pts = tris.reshape(-1, 3)
    rounded = np.round(pts, decimals=12)  # Handle floating point precision
    uniq, inv = np.unique(rounded, axis=0, return_inverse=True)
    faces_idx = inv.reshape(-1, 3)
    
    # VTK faces format: [3, i, j, k, 3, i, j, k, ...]
    faces = np.hstack([np.concatenate(([3], tri)) for tri in faces_idx]).astype(np.int64)
    mesh = pv.PolyData(uniq, faces)
    mesh.clean(inplace=True)
    return mesh

def add_ray_paths(plotter: pv.Plotter, paths: List[np.ndarray], 
                  line_width: float = 2.0, color="crimson", opacity: float = 0.9):
    """Add multiple ray paths to the plotter."""
    for pts in paths:
        if len(pts) > 1:
            line = pv.lines_from_points(pts, close=False)
            plotter.add_mesh(line, color=color, line_width=line_width, opacity=opacity)

def make_intensity_grid_xy(xy: np.ndarray, z0: float,
                           bins: int = 200,
                           margin: float = 1.2,
                           extent_xy: Optional[Tuple[Tuple[float,float], Tuple[float,float]]] = None) -> Tuple[pv.ImageData, str]:
    """
    Create 2D intensity histogram as PyVista ImageData for visualization.
    
    Parameters:
    - xy: (N, 2) intersection points
    - z0: z-coordinate of the plane
    - bins: histogram resolution
    - margin: expansion factor for auto-computed extents
    - extent_xy: manual extent specification ((xmin,xmax), (ymin,ymax))
    
    Returns (grid, scalar_name) tuple.
    """
    def _empty_grid():
        """Create empty grid for cases with no data."""
        grid = pv.ImageData()
        grid.dimensions = (2, 2, 2)
        grid.origin = (0.0, 0.0, z0 - 1e-6)
        grid.spacing = (1e-3, 1e-3, 2e-6)
        grid.cell_data['I'] = np.zeros((1,), dtype=float)
        return grid, 'I'

    if xy.size == 0:
        return _empty_grid()

    if isinstance(bins, int):
        nx = ny = max(8, bins)
    else:
        nx, ny = bins

    # Determine histogram extents
    if extent_xy is None:
        xmin, xmax = xy[:,0].min(), xy[:,0].max()
        ymin, ymax = xy[:,1].min(), xy[:,1].max()
        cx, cy = 0.5*(xmin+xmax), 0.5*(ymin+ymax)
        sx, sy = (xmax-xmin), (ymax-ymin)
        sx = sx if sx > 0 else 1e-6
        sy = sy if sy > 0 else 1e-6
        halfx = 0.5*margin*sx
        halfy = 0.5*margin*sy
        xmin, xmax = cx - halfx, cx + halfx
        ymin, ymax = cy - halfy, cy + halfy
    else:
        (xmin, xmax), (ymin, ymax) = extent_xy

    # Create 2D histogram
    H, x_edges, y_edges = np.histogram2d(xy[:,0], xy[:,1], 
                                        bins=[nx, ny], 
                                        range=[[xmin, xmax], [ymin, ymax]])

    # Build PyVista ImageData
    grid = pv.ImageData()
    grid.dimensions = (nx + 1, ny + 1, 2)  # Points dimensions
    dx = (x_edges[-1] - x_edges[0]) / nx
    dy = (y_edges[-1] - y_edges[0]) / ny
    grid.origin = (x_edges[0], y_edges[0], z0 - 1e-6)
    grid.spacing = (dx, dy, 2e-6)

    # Add intensity data (VTK expects Fortran order)
    grid.cell_data['I'] = H.T.ravel(order='F')
    grid.set_active_scalars('I')
    return grid, 'I'

def add_intensity_plane(plotter: pv.Plotter, grid: pv.DataSet, 
                       scalar_name: str = 'I', cmap: str = 'inferno', 
                       opacity: float = 0.85):
    """Add intensity heatmap to the plotter."""
    plotter.add_mesh(grid, scalars=scalar_name, cmap=cmap, 
                    opacity=opacity, lighting=False, show_edges=False)

def create_optical_scene(mesh_polydata: pv.PolyData, 
                        ray_paths: List[np.ndarray],
                        title: str = "Optical Ray Tracing",
                        intensity_planes: Optional[List[Tuple[pv.DataSet, str]]] = None,
                        window_size: Tuple[int, int] = (900, 700)) -> pv.Plotter:
    """
    Create complete optical visualization scene.
    
    Parameters:
    - mesh_polydata: optical element mesh
    - ray_paths: list of ray path arrays
    - title: plot title
    - intensity_planes: list of (grid, scalar_name) tuples for intensity maps
    - window_size: plotter window size
    
    Returns configured PyVista plotter.
    """
    p = pv.Plotter(window_size=window_size)
    
    # Add title
    p.add_title(title, font_size=12)
    
    # Add optical element mesh
    p.add_mesh(mesh_polydata, color="lightblue", opacity=0.35, 
              show_edges=True, lighting=True, smooth_shading=True)
    
    # Add ray paths
    add_ray_paths(p, ray_paths)
    
    # Add intensity planes if provided
    if intensity_planes:
        for grid, scalar_name in intensity_planes:
            add_intensity_plane(p, grid, scalar_name=scalar_name)
    
    # Add coordinate axes and grid
    p.add_axes(interactive=True)
    p.enable_parallel_projection()  # Orthographic view
    p.show_grid()
    
    return p
