#!/usr/bin/env python3
"""
Analysis utilities for ray tracing results.
Includes intensity analysis, beam parameter fitting, and cross-sectional analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict

def intersect_path_with_z(path: np.ndarray, z0: float) -> Optional[np.ndarray]:
    """Find intersection of ray path with plane z=z0."""
    z = path[:, 2]
    dz = z[1:] - z[:-1]
    # Find segments that cross z0
    crosses = np.where((z[:-1] - z0) * (z[1:] - z0) <= 0)[0]
    for i in crosses:
        if dz[i] == 0:
            continue
        t = (z0 - z[i]) / dz[i]
        if 0.0 <= t <= 1.0:
            p = path[i] + t * (path[i+1] - path[i])
            return p
    return None

def intersect_paths_with_z(paths: List[np.ndarray], z0: float) -> np.ndarray:
    """Find intersections of multiple ray paths with plane z=z0."""
    pts = []
    for path in paths:
        p = intersect_path_with_z(path, z0)
        if p is not None:
            pts.append([p[0], p[1]])
    return np.asarray(pts, dtype=float) if pts else np.empty((0, 2), dtype=float)

def fit_gaussian_2d(xy: np.ndarray) -> Dict:
    """
    Fit 2D Gaussian to point distribution.
    Returns parameters: mu, Sigma, beam radii, rotation angle.
    """
    if xy.size == 0:
        return {}
    
    mu = xy.mean(axis=0)
    X = xy - mu
    # Sample covariance matrix
    Sigma = (X.T @ X) / max(1, xy.shape[0]-1)
    
    # Eigendecomposition for principal axes
    evals, evecs = np.linalg.eigh(Sigma)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    sigmas = np.sqrt(np.maximum(evals, 0.0))
    w = np.sqrt(2.0) * sigmas  # 1/e^2 radii
    
    # Rotation angle of major axis
    angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
    
    return {
        'mu': mu,
        'Sigma': Sigma,
        'eigvals': evals,
        'eigvecs': evecs,
        'sigmas': sigmas,
        'w': w,
        'angle_deg': angle,
    }

def ellipse_points(mu, axes_w, angle_deg, n=200):
    """Generate points for 1/e^2 ellipse contour."""
    t = np.linspace(0, 2*np.pi, n)
    ca, sa = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    R = np.array([[ca, -sa], [sa, ca]])
    pts = (R @ (np.vstack((axes_w[0]*np.cos(t), axes_w[1]*np.sin(t)))))
    pts = pts.T + mu
    return pts

def plot_beam_cross_section(xy: np.ndarray, z0: float, fit: bool = True, 
                           bins: int = 0, title_prefix: str = ""):
    """
    Plot beam cross-section at specified z-plane.
    
    Parameters:
    - xy: (N,2) array of intersection points
    - z0: z-coordinate of analysis plane
    - fit: whether to fit and display Gaussian parameters
    - bins: number of histogram bins for background (0 = no histogram)
    - title_prefix: prefix for plot title
    """
    if xy.size == 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_title(f"{title_prefix} z={z0:.4g} m — no intersections")
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_aspect('equal', adjustable='box')
        plt.show()
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    
    # Optional background histogram
    if bins and bins > 0:
        H, xedges, yedges = np.histogram2d(xy[:,0], xy[:,1], bins=bins)
        ax.imshow(H.T, origin='lower', 
                 extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                 cmap='Greys', alpha=0.35, aspect='equal', interpolation='nearest')

    # Scatter plot of ray intersections
    ax.scatter(xy[:,0], xy[:,1], s=6, alpha=0.7, color='blue')

    txt = ""
    if fit:
        pars = fit_gaussian_2d(xy)
        if pars:
            mu = pars['mu']
            w = pars['w']
            ang = pars['angle_deg']
            
            # Plot 1/e^2 ellipse
            ell = ellipse_points(mu, w, ang)
            ax.plot(ell[:,0], ell[:,1], 'r-', lw=2, label='1/e² contour')
            
            txt = (f"μ=({mu[0]:.3g},{mu[1]:.3g})  "
                  f"w=(w_x={w[0]*1e3:.2f} mm, w_y={w[1]*1e3:.2f} mm)  "
                  f"angle={ang:.1f}°")

    ax.set_title(f"{title_prefix} z={z0:.4g} m  {txt}")
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    if fit and txt:
        ax.legend()
    plt.tight_layout()
    plt.show()

def analyze_beam_evolution(paths: List[np.ndarray], z_positions: np.ndarray) -> Dict:
    """
    Analyze beam evolution along propagation axis.
    
    Returns dictionary with z-positions and corresponding beam parameters.
    """
    results = {
        'z': z_positions,
        'w_x': [],
        'w_y': [],
        'mu_x': [],
        'mu_y': [],
        'n_rays': []
    }
    
    for z0 in z_positions:
        xy = intersect_paths_with_z(paths, z0)
        results['n_rays'].append(len(xy))
        
        if len(xy) > 0:
            pars = fit_gaussian_2d(xy)
            if pars:
                results['w_x'].append(pars['w'][0])
                results['w_y'].append(pars['w'][1])
                results['mu_x'].append(pars['mu'][0])
                results['mu_y'].append(pars['mu'][1])
            else:
                results['w_x'].append(0)
                results['w_y'].append(0)
                results['mu_x'].append(0)
                results['mu_y'].append(0)
        else:
            results['w_x'].append(0)
            results['w_y'].append(0)
            results['mu_x'].append(0)
            results['mu_y'].append(0)
    
    # Convert to arrays
    for key in ['w_x', 'w_y', 'mu_x', 'mu_y', 'n_rays']:
        results[key] = np.array(results[key])
    
    return results

def plot_beam_evolution(results: Dict, title: str = "Beam Evolution"):
    """Plot beam size evolution along propagation axis."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    z = results['z']
    
    # Beam size evolution
    ax1.plot(z, results['w_x'] * 1e3, 'b-', label='w_x', linewidth=2)
    ax1.plot(z, results['w_y'] * 1e3, 'r-', label='w_y', linewidth=2)
    ax1.set_xlabel('z [m]')
    ax1.set_ylabel('Beam radius [mm]')
    ax1.set_title(f'{title} - Beam Size Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Beam center evolution
    ax2.plot(z, results['mu_x'] * 1e3, 'b-', label='μ_x', linewidth=2)
    ax2.plot(z, results['mu_y'] * 1e3, 'r-', label='μ_y', linewidth=2)
    ax2.set_xlabel('z [m]')
    ax2.set_ylabel('Beam center [mm]')
    ax2.set_title(f'{title} - Beam Center Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
