#!/usr/bin/env python3
"""
Mathematical utilities for ray tracing operations.
Includes vector operations, normalization, and optical physics calculations.
"""

import numpy as np
import numba as nb
from typing import Tuple, Optional

EPS = 1e-7

# =========================
# Numba-accelerated functions
# =========================

@nb.njit(cache=True, fastmath=True)
def _normalize3(v):
    """Fast 3D vector normalization with Numba."""
    n = np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if n > 0.0:
        return v / n
    return v

@nb.njit(cache=True, fastmath=True)
def refract_one_sided_nb(I, N, n1, n2):
    """
    Numba-accelerated Snell's law refraction.
    N points into destination medium n2.
    Returns (refracted_direction, total_internal_reflection_flag).
    """
    I = _normalize3(I)
    N = _normalize3(N)
    cosi = -(I[0]*N[0] + I[1]*N[1] + I[2]*N[2])
    if cosi < -1.0:
        cosi = -1.0
    elif cosi > 1.0:
        cosi = 1.0
    eta = n1 / n2
    k = 1.0 - eta*eta*(1.0 - cosi*cosi)
    if k < 0.0:
        return np.zeros(3), True
    t0 = eta*I[0] + (eta*cosi - np.sqrt(k))*N[0]
    t1 = eta*I[1] + (eta*cosi - np.sqrt(k))*N[1]
    t2 = eta*I[2] + (eta*cosi - np.sqrt(k))*N[2]
    T = np.array((t0, t1, t2))
    T = _normalize3(T)
    return T, False

@nb.njit(cache=True, fastmath=True)
def reflect_nb(I, N):
    """Numba-accelerated reflection calculation."""
    I = _normalize3(I)
    N = _normalize3(N)
    dotIN = I[0]*N[0] + I[1]*N[1] + I[2]*N[2]
    R = np.array((I[0] - 2.0*dotIN*N[0],
                  I[1] - 2.0*dotIN*N[1],
                  I[2] - 2.0*dotIN*N[2]))
    return _normalize3(R)

@nb.njit(cache=True, fastmath=True)
def mt_closest_intersection(o, d, V0, E1, E2):
    """
    Numba-accelerated Möller-Trumbore ray-triangle intersection.
    Returns (triangle_index, distance) or (-1, large_value) if no hit.
    """
    best_t = 1.0e30
    best_i = -1
    for i in range(V0.shape[0]):
        v0 = V0[i]
        e1 = E1[i]
        e2 = E2[i]
        # pvec = cross(d, e2)
        p0 = d[1]*e2[2] - d[2]*e2[1]
        p1 = d[2]*e2[0] - d[0]*e2[2]
        p2 = d[0]*e2[1] - d[1]*e2[0]
        det = e1[0]*p0 + e1[1]*p1 + e1[2]*p2
        if det > -1e-7 and det < 1e-7:
            continue
        inv_det = 1.0 / det
        tvec0 = o[0] - v0[0]
        tvec1 = o[1] - v0[1]
        tvec2 = o[2] - v0[2]
        u = (tvec0*p0 + tvec1*p1 + tvec2*p2) * inv_det
        if u < 0.0 or u > 1.0:
            continue
        # qvec = cross(tvec, e1)
        q0 = tvec1*e1[2] - tvec2*e1[1]
        q1 = tvec2*e1[0] - tvec0*e1[2]
        q2 = tvec0*e1[1] - tvec1*e1[0]
        v = (d[0]*q0 + d[1]*q1 + d[2]*q2) * inv_det
        if v < 0.0 or (u + v) > 1.0:
            continue
        t = (e2[0]*q0 + e2[1]*q1 + e2[2]*q2) * inv_det
        if t > 1e-7 and t < best_t:
            best_t = t
            best_i = i
    return best_i, best_t

# =========================
# Standard Python functions
# =========================

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def reflect(I: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Reflect incident ray I off surface with normal N."""
    return normalize(I - 2.0 * np.dot(I, N) * N)

def refract_one_sided(I: np.ndarray, N_to_n2: np.ndarray, n1: float, n2: float) -> Tuple[Optional[np.ndarray], bool]:
    """
    Snell refraction assuming N_to_n2 points INTO the destination medium (n2).
    Returns (refracted_direction, total_internal_reflection_flag).
    """
    I = I / np.linalg.norm(I)
    N = N_to_n2 / np.linalg.norm(N_to_n2)
    cosi = -np.clip(np.dot(I, N), -1.0, 1.0)  # expect >= 0 when N points into n2
    eta = n1 / n2
    k = 1.0 - eta*eta*(1.0 - cosi*cosi)
    if k < 0.0:
        return None, True
    T = eta*I + (eta*cosi - np.sqrt(k))*N
    T /= np.linalg.norm(T)
    return T, False

def orthonormal_frame_from_axis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unit vectors (u, v, w) with w || axis."""
    w = normalize(np.asarray(axis, dtype=float))
    # pick a vector not parallel to w
    a = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = normalize(np.cross(a, w))
    v = normalize(np.cross(w, u))
    return u, v, w
