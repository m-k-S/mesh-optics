#!/usr/bin/env python3
"""
Geometric primitives and mesh generation for optical elements.
Focuses on lens geometry without sphere generation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from math_utils import normalize, EPS

@dataclass
class Triangle:
    """A triangle defined by three vertices and outward normal."""
    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    normal_out: np.ndarray

    @staticmethod
    def from_vertices(v0, v1, v2):
        """Create triangle from vertices with computed outward normal."""
        v0 = np.asarray(v0, dtype=float)
        v1 = np.asarray(v1, dtype=float)
        v2 = np.asarray(v2, dtype=float)
        n = normalize(np.cross(v1 - v0, v2 - v0))  # outward via right-hand rule
        return Triangle(v0, v1, v2, n)

    def intersect(self, ray_o: np.ndarray, ray_d: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Möller–Trumbore ray-triangle intersection. Returns (t, u, v) or None."""
        v0v1 = self.v1 - self.v0
        v0v2 = self.v2 - self.v0
        pvec = np.cross(ray_d, v0v2)
        det = np.dot(v0v1, pvec)
        if abs(det) < EPS:
            return None
        inv_det = 1.0 / det
        tvec = ray_o - self.v0
        u = np.dot(tvec, pvec) * inv_det
        if u < 0.0 or u > 1.0:
            return None
        qvec = np.cross(tvec, v0v1)
        v = np.dot(ray_d, qvec) * inv_det
        if v < 0.0 or (u + v) > 1.0:
            return None
        t = np.dot(v0v2, qvec) * inv_det
        if t > EPS:
            return (t, u, v)
        return None

@dataclass
class Mesh:
    """A mesh composed of triangles with a refractive index."""
    triangles: List[Triangle]
    n_inside: float
    name: str = "mesh"

    @staticmethod
    def from_triangle_array(tris_xyz: np.ndarray, n_inside: float, name: str = "mesh"):
        """Create mesh from array of triangle vertices."""
        triangles = [Triangle.from_vertices(*tris_xyz[i]) for i in range(tris_xyz.shape[0])]
        return Mesh(triangles=triangles, n_inside=n_inside, name=name)

def plano_convex_tris(
    aperture_radius: float,
    R: float,
    center_thickness: float,
    radial_segments: int = 48,
    azimuth_segments: int = 180,
    center=(0.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Generate triangulated plano-convex/concave lens mesh.
    
    Parameters:
    - aperture_radius: lens radius
    - R: radius of curvature (R > 0 = convex toward +Z, R < 0 = concave)
    - center_thickness: thickness at center
    - radial_segments: radial discretization
    - azimuth_segments: azimuthal discretization
    - center: lens center position
    
    Returns array of triangles with shape (N, 3, 3) with outward normals.
    """
    if radial_segments < 2 or azimuth_segments < 8:
        raise ValueError("radial_segments >= 2 and azimuth_segments >= 8 are recommended.")

    cx, cy, cz = center
    z_plane = cz - center_thickness * 0.5   # flat face
    z_vertex = cz + center_thickness * 0.5   # spherical vertex

    a = float(aperture_radius)
    if a <= 0:
        raise ValueError("aperture_radius must be > 0.")

    Rabs = abs(R)
    if a >= Rabs:
        raise ValueError("aperture_radius must be < |R| for a valid spherical surface.")

    # Sphere center location relative to the vertex
    zc = z_vertex - R

    def z_sphere(r: np.ndarray) -> np.ndarray:
        """Calculate z-coordinate on spherical surface."""
        root = np.sqrt(Rabs*Rabs - r*r)
        if R >= 0:
            # convex toward +Z: vertex is the highest point; edge is lower
            return zc + root
        else:
            # concave toward +Z: vertex is the lowest point; edge is higher
            return zc - root

    # Validate geometry
    edge_z = float(z_sphere(np.array([a]))[0])
    if edge_z <= z_plane:
        raise ValueError(
            f"Invalid geometry: spherical edge z={edge_z:.6g} <= plane z={z_plane:.6g}. "
            "Decrease aperture, increase |R|, or increase center_thickness."
        )

    # Build polar grids
    rs = np.linspace(0.0, a, radial_segments + 1)
    thetas = np.linspace(0.0, 2.0*np.pi, azimuth_segments, endpoint=False)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)

    def ring_xy(r):
        """Generate x,y coordinates for a ring at radius r."""
        return (cx + r*cos_t, cy + r*sin_t)

    # Generate spherical surface points
    sph_pts = []
    for r in rs:
        x, y = ring_xy(r)
        z = np.full_like(x, z_sphere(np.array([r]))[0])
        sph_pts.append(np.stack([x, y, z], axis=-1))
    sph_pts = np.stack(sph_pts, axis=0)

    # Generate planar surface points
    plane_pts = []
    for r in rs:
        x, y = ring_xy(r)
        z = np.full_like(x, z_plane)
        plane_pts.append(np.stack([x, y, z], axis=-1))
    plane_pts = np.stack(plane_pts, axis=0)

    tris = []
    
    def add_quad(v00, v10, v11, v01, outward_ccw=True):
        """Add a quadrilateral as two triangles with proper winding."""
        if outward_ccw:
            tris.append(np.array([v00, v10, v11], dtype=float))
            tris.append(np.array([v00, v11, v01], dtype=float))
        else:
            tris.append(np.array([v00, v11, v10], dtype=float))
            tris.append(np.array([v00, v01, v11], dtype=float))

    # Spherical cap (outward normals)
    for j in range(radial_segments):
        for k in range(azimuth_segments):
            k2 = (k + 1) % azimuth_segments
            v00 = sph_pts[j,   k ]
            v10 = sph_pts[j+1, k ]
            v11 = sph_pts[j+1, k2]
            v01 = sph_pts[j,   k2]
            add_quad(v00, v10, v11, v01, outward_ccw=True)

    # Plane (outward = -Z): clockwise when viewed from +Z
    for j in range(radial_segments):
        for k in range(azimuth_segments):
            k2 = (k + 1) % azimuth_segments
            v00 = plane_pts[j,   k ]
            v10 = plane_pts[j+1, k ]
            v11 = plane_pts[j+1, k2]
            v01 = plane_pts[j,   k2]
            add_quad(v00, v10, v11, v01, outward_ccw=False)

    # Cylindrical rim (outward radial)
    j = radial_segments
    for k in range(azimuth_segments):
        k2 = (k + 1) % azimuth_segments
        v_plane_k  = plane_pts[j, k]
        v_plane_k2 = plane_pts[j, k2]
        v_sph_k    = sph_pts[j, k]
        v_sph_k2   = sph_pts[j, k2]
        add_quad(v_plane_k, v_plane_k2, v_sph_k2, v_sph_k, outward_ccw=True)

    return np.asarray(tris, dtype=float)
