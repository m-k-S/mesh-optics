#!/usr/bin/env python3
"""
Ray source generation for optical simulations.
Includes point sources, parallel beams, and Gaussian beam generation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from math_utils import normalize, orthonormal_frame_from_axis

@dataclass
class Ray:
    """A ray with origin, direction, wavelength, and intensity."""
    origin: np.ndarray
    direction: np.ndarray
    wavelength_nm: float = 550.0
    intensity: float = 1.0

def point_source_single(origin: np.ndarray, theta_deg: float, phi_deg: float, wavelength_nm: float = 550.0) -> Ray:
    """Generate a single ray from a point source at specified angles."""
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    d = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ], dtype=float)
    return Ray(origin=np.asarray(origin, dtype=float), direction=normalize(d), wavelength_nm=wavelength_nm)

def parallel_beam_single(point_on_plane: np.ndarray, direction: np.ndarray, wavelength_nm: float = 550.0) -> Ray:
    """Generate a single ray from a parallel beam."""
    return Ray(origin=np.asarray(point_on_plane, dtype=float),
               direction=normalize(np.asarray(direction, dtype=float)),
               wavelength_nm=wavelength_nm)

def _gaussian_xy(n: int, w_at_plane: float, rng: np.random.Generator):
    """
    Sample (x,y) from 2D Gaussian with I(r) ~ exp(-2 r^2 / w^2).
    That's equivalent to x,y ~ N(0, sigma^2) with sigma = w / sqrt(2).
    """
    sigma = w_at_plane / np.sqrt(2.0)
    x = rng.normal(0.0, sigma, size=n)
    y = rng.normal(0.0, sigma, size=n)
    return x, y

def gaussian_beam_rays(
    n_rays: int,
    waist_radius: float,           # w0  (1/e^2 intensity radius) at the waist
    wavelength: float,             # meters
    waist_z: float,                # z position of the waist along the beam axis
    launch_z: float,               # z position of the launch plane (where rays start)
    axis: np.ndarray = np.array([0.0, 0.0, 1.0]),
    center_xy: np.ndarray = np.array([0.0, 0.0]),  # beam center in the launch plane
    rng: Optional[np.random.Generator] = None,
) -> List[Ray]:
    """
    Generate a bundle of rays that sample a fundamental Gaussian beam at launch_z.
    Ray directions follow the local Gaussian wavefront curvature.
    
    Parameters:
    - n_rays: number of rays to generate
    - waist_radius: 1/e^2 intensity radius at the beam waist
    - wavelength: wavelength in meters
    - waist_z: z-position of the beam waist
    - launch_z: z-position where rays are launched
    - axis: beam propagation axis (default +z)
    - center_xy: beam center in the launch plane
    - rng: random number generator
    
    Returns list of Ray objects with proper Gaussian beam characteristics.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Gaussian beam parameters
    w0 = float(waist_radius)
    zR = np.pi * w0 * w0 / float(wavelength)           # Rayleigh range
    z = float(launch_z) - float(waist_z)               # distance from waist to plane
    w_z = w0 * np.sqrt(1.0 + (z / zR) ** 2)            # beam radius at plane

    # Wavefront radius R(z) (positive for forward +z propagation)
    if abs(z) < 1e-15:
        Rz = np.inf
    else:
        Rz = z * (1.0 + (zR / z) ** 2)

    # Build coordinate frame and center point at the plane
    u, v, w = orthonormal_frame_from_axis(axis)
    origin_plane = center_xy[0] * u + center_xy[1] * v + (launch_z * w)

    # Sample transverse positions with Gaussian distribution
    xs, ys = _gaussian_xy(n_rays, w_z, rng)

    rays: List[Ray] = []
    # Sphere center for the wavefront curvature
    wavefront_center = origin_plane + (Rz * w) if np.isfinite(Rz) else None

    for x, y in zip(xs, ys):
        p = origin_plane + x * u + y * v
        if wavefront_center is None:
            # Collimated beam (at waist)
            d = w.copy()
        else:
            # Converging/diverging beam
            d = normalize(wavefront_center - p)
        rays.append(Ray(origin=p, direction=d))
    
    return rays
