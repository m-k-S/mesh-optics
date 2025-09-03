#!/usr/bin/env python3
# Optional speed-ups: install numba (`pip install numba`). The script will JIT-accelerate intersections/refractions
# and traverse triangles using a struct-of-arrays layout. For many rays, it also parallelizes tracing.
# pv_raytrace_demo.py
# Interactive (PyVista/VTK) visualization of a refracting ray through a triangle-mesh object.

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pyvista as pv
import numba as nb
import matplotlib.pyplot as plt


# =========================
# Math helpers
# =========================
EPS = 1e-7

# --- Numba-accelerated math helpers (if numba is available) ---
@nb.njit(cache=True, fastmath=True)
def _normalize3(v):
    n = np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if n > 0.0:
        return v / n
    return v

@nb.njit(cache=True, fastmath=True)
def refract_one_sided_nb(I, N, n1, n2):
    # N points into destination medium n2; vectors are length-agnostic here
    # Normalize locally for speed/robustness
    I = _normalize3(I)
    N = _normalize3(N)
    cosi = - (I[0]*N[0] + I[1]*N[1] + I[2]*N[2])
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
    I = _normalize3(I)
    N = _normalize3(N)
    dotIN = I[0]*N[0] + I[1]*N[1] + I[2]*N[2]
    R = np.array((I[0] - 2.0*dotIN*N[0],
                      I[1] - 2.0*dotIN*N[1],
                      I[2] - 2.0*dotIN*N[2]))
    return _normalize3(R)

@nb.njit(cache=True, fastmath=True)
def mt_closest_intersection(o, d, V0, E1, E2):
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

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def reflect(I: np.ndarray, N: np.ndarray) -> np.ndarray:
    return normalize(I - 2.0 * np.dot(I, N) * N)

def refract(I: np.ndarray, N: np.ndarray, n1: float, n2: float) -> Tuple[Optional[np.ndarray], bool]:
    """
    Snell refraction with TIR. I,N are unit vectors; N is geometric outward normal.
    Returns (T, tir_flag).
    """
    I = normalize(I)
    N = normalize(N)
    cosi = np.clip(np.dot(I, N), -1.0, 1.0)
    etai, etat = n1, n2
    n = N.copy()
    if cosi > 0:
        n = -N
        etai, etat = etat, etai
        cosi = -cosi
    eta = etai / etat
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    if k < 0.0:
        return None, True
    T = normalize(eta * I + (eta * cosi - np.sqrt(k)) * n)
    return T, False

def refract_one_sided(I: np.ndarray, N_to_n2: np.ndarray, n1: float, n2: float):
    """
    Snell refraction assuming N_to_n2 points INTO the destination medium (n2).
    Returns (T, tir_flag). No internal swapping of indices or normals.
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

# =========================
# Geometry & Scene
# =========================
@dataclass
class Triangle:
    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    normal_out: np.ndarray

    @staticmethod
    def from_vertices(v0, v1, v2):
        v0 = np.asarray(v0, dtype=float)
        v1 = np.asarray(v1, dtype=float)
        v2 = np.asarray(v2, dtype=float)
        n = normalize(np.cross(v1 - v0, v2 - v0))  # outward via right-hand rule
        return Triangle(v0, v1, v2, n)

    def intersect(self, ray_o: np.ndarray, ray_d: np.ndarray) -> Optional[Tuple[float, float, float]]:
        # Möller–Trumbore
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
    triangles: List[Triangle]
    n_inside: float
    name: str = "mesh"

    @staticmethod
    def from_triangle_array(tris_xyz: np.ndarray, n_inside: float, name: str = "mesh"):
        triangles = [Triangle.from_vertices(*tris_xyz[i]) for i in range(tris_xyz.shape[0])]
        return Mesh(triangles=triangles, n_inside=n_inside, name=name)

@dataclass
class Ray:
    origin: np.ndarray
    direction: np.ndarray
    wavelength_nm: float = 550.0
    intensity: float = 1.0

@dataclass
class Hit:
    t: float
    point: np.ndarray
    normal_out: np.ndarray
    mesh: Mesh
    tri_index: int

class Scene:
    def __init__(self, meshes: List[Mesh], n_outside: float = 1.0):
        self.meshes = meshes
        self.n_outside = n_outside
        # Optional triangle arrays for accelerated intersection
        self._V0 = None
        self._E1 = None
        self._E2 = None
        self._Nout = None
        self._tri_mesh_idx = None

    def build_accel(self):
        V0 = []
        E1 = []
        E2 = []
        Nout = []
        tri_mesh_idx = []
        for mi, mesh in enumerate(self.meshes):
            for tri in mesh.triangles:
                V0.append(tri.v0)
                E1.append(tri.v1 - tri.v0)
                E2.append(tri.v2 - tri.v0)
                Nout.append(tri.normal_out)
                tri_mesh_idx.append(mi)
        if len(V0) == 0:
            return
        self._V0 = np.asarray(V0, dtype=np.float64)
        self._E1 = np.asarray(E1, dtype=np.float64)
        self._E2 = np.asarray(E2, dtype=np.float64)
        self._Nout = np.asarray(Nout, dtype=np.float64)
        self._tri_mesh_idx = np.asarray(tri_mesh_idx, dtype=np.int64)

    def closest_intersection(self, o: np.ndarray, d: np.ndarray) -> Optional[Hit]:
        # Fast path: Numba-accelerated array traversal if available
        if nb is not None and self._V0 is not None:
            idx, t = mt_closest_intersection(o, d, self._V0, self._E1, self._E2)
            if idx == -1:
                return None
            p = o + d * t
            normal_out = self._Nout[idx]
            mesh = self.meshes[int(self._tri_mesh_idx[idx])]
            return Hit(t=t, point=p, normal_out=normal_out, mesh=mesh, tri_index=int(idx))
        # Fallback: Python loop
        best = None
        best_t = np.inf
        for mesh in self.meshes:
            for i, tri in enumerate(mesh.triangles):
                res = tri.intersect(o, d)
                if res is None:
                    continue
                t, _, _ = res
                if t < best_t and t > EPS:
                    p = o + t * d
                    best = Hit(t=t, point=p, normal_out=tri.normal_out, mesh=mesh, tri_index=i)
                    best_t = t
        return best

# =========================
# Tracer (single ray)
# =========================
def trace_single_ray(scene: Scene,
                     ray: Ray,
                     max_bounces: int = 100,
                     max_path_length: float = 1e6) -> List[np.ndarray]:
    pts = [ray.origin.copy()]
    o = ray.origin.copy()
    d = normalize(ray.direction.copy())

    # stack of (mesh_or_None, n). Start outside.
    med_stack: List[Tuple[Optional[Mesh], float]] = [(None, scene.n_outside)]

    path_len = 0.0
    bounces = 0
    while bounces < max_bounces and path_len < max_path_length:
        hit = scene.closest_intersection(o, d)
        if hit is None:
            pts.append(o + d * (min(200.0, max_path_length - path_len)))
            break

        p = hit.point
        path_len += np.linalg.norm(p - o)
        pts.append(p.copy())

        top_mesh, n1 = med_stack[-1]
        entering = (top_mesh is not hit.mesh)
        # destination index:
        n2 = hit.mesh.n_inside if entering else (med_stack[-2][1] if len(med_stack) > 1 else scene.n_outside)

        # orient the normal so it points INTO n2
        N_to_n2 = hit.normal_out if entering else (-hit.normal_out)

        # refraction or reflection
        if nb is not None:
            T, tir = refract_one_sided_nb(d, N_to_n2, n1, n2)
            if tir:
                d = reflect_nb(d, hit.normal_out)
            else:
                d = T
                if entering:
                    med_stack.append((hit.mesh, hit.mesh.n_inside))      # now inside lens
                else:
                    if len(med_stack) > 1:
                        med_stack.pop()                                  # back to previous medium
        else:
            T, tir = refract_one_sided(d, N_to_n2, n1, n2)
            if tir:
                d = reflect(d, hit.normal_out)                           # stay in same medium
            else:
                d = T
                if entering:
                    med_stack.append((hit.mesh, hit.mesh.n_inside))      # now inside lens
                else:
                    if len(med_stack) > 1:
                        med_stack.pop()                                  # back to previous medium

        o = p + d * (10 * EPS)  # nudge off surface
        bounces += 1

    return pts

# --- Parallel ray-bundle tracer ---
from concurrent.futures import ThreadPoolExecutor, as_completed

def trace_ray_bundle_parallel(scene: Scene, rays: List[Ray], max_bounces: int = 100, max_path_length: float = 1e6, workers: Optional[int] = None) -> List[np.ndarray]:
    paths = [None] * len(rays)
    # Ensure accelerator is built once
    if hasattr(scene, 'build_accel'):
        scene.build_accel()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(trace_single_ray, scene, r, max_bounces, max_path_length): i for i, r in enumerate(rays)}
        for f in as_completed(futs):
            i = futs[f]
            pts = f.result()
            paths[i] = np.asarray(pts)
    return paths

def trace_ray_bundle(scene: Scene, rays: List[Ray],
                     max_bounces: int = 100, max_path_length: float = 1e6) -> List[np.ndarray]:
    """Trace many rays and return a list of polylines (each polyline is (M_i,3))."""
    paths = []
    for r in rays:
        pts = trace_single_ray(scene, r, max_bounces=max_bounces, max_path_length=max_path_length)
        paths.append(np.asarray(pts))
    return paths

# =========================
# Mesh builders
# =========================
def icosahedron(radius=1.0):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=float)
    verts = np.array([normalize(v) * radius for v in verts])
    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=int)
    return verts, faces

def _midpoint_cache(mid_cache, verts, i, j, radius):
    key = tuple(sorted((i, j)))
    if key in mid_cache:
        return mid_cache[key], verts
    v = normalize((verts[i] + verts[j]) * 0.5) * radius
    verts = np.vstack([verts, v])
    idx = len(verts) - 1
    mid_cache[key] = idx
    return idx, verts

def subdivide(verts, faces, radius, levels=2):
    for _ in range(levels):
        mid_cache = {}
        new_faces = []
        verts_list = verts
        for tri in faces:
            i0, i1, i2 = tri
            m01, verts_list = _midpoint_cache(mid_cache, verts_list, i0, i1, radius)
            m12, verts_list = _midpoint_cache(mid_cache, verts_list, i1, i2, radius)
            m20, verts_list = _midpoint_cache(mid_cache, verts_list, i2, i0, radius)
            new_faces.extend([
                [i0, m01, m20],
                [i1, m12, m01],
                [i2, m20, m12],
                [m01, m12, m20],
            ])
        verts = verts_list
        faces = np.array(new_faces, dtype=int)
    return verts, faces

def make_icosphere_tris(radius=0.5, levels=3, center=(0,0,0)):
    v, f = icosahedron(radius=radius)
    v, f = subdivide(v, f, radius=radius, levels=levels)
    v = v + np.asarray(center, dtype=float)
    tris = np.asarray([[v[i], v[j], v[k]] for i, j, k in f], dtype=float)
    return tris

def plano_convex_tris(
    aperture_radius: float,
    R: float,
    center_thickness: float,
    radial_segments: int = 48,
    azimuth_segments: int = 180,
    center=(0.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Watertight plano-convex/concave lens: flat face at z = -ct/2, spherical vertex at z = +ct/2.
    Sign convention: R > 0 => convex toward +Z; R < 0 => concave toward +Z.
    Returns tris of shape (N,3,3) with outward winding.
    """
    if radial_segments < 2 or azimuth_segments < 8:
        raise ValueError("radial_segments >= 2 and azimuth_segments >= 8 are recommended.")

    cx, cy, cz = center
    z_plane  = cz - center_thickness * 0.5   # flat face
    z_vertex = cz + center_thickness * 0.5   # spherical vertex

    a = float(aperture_radius)
    if a <= 0:
        raise ValueError("aperture_radius must be > 0.")

    Rabs = abs(R)
    if a >= Rabs:
        raise ValueError("aperture_radius must be < |R| for a valid spherical surface.")

    # Correct sphere center location relative to the vertex
    zc = z_vertex - R  # << FIXED

    def z_sphere(r: np.ndarray) -> np.ndarray:
        root = np.sqrt(Rabs*Rabs - r*r)
        if R >= 0:
            # convex toward +Z: vertex is the highest point; edge is lower
            return zc + root
        else:
            # concave toward +Z: vertex is the lowest point; edge is higher
            return zc - root

    # Geometry sanity: spherical rim should not pass below the plane
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
        return (cx + r*cos_t, cy + r*sin_t)

    sph_pts = []
    for r in rs:
        x, y = ring_xy(r)
        z = np.full_like(x, z_sphere(np.array([r]))[0])
        sph_pts.append(np.stack([x, y, z], axis=-1))
    sph_pts = np.stack(sph_pts, axis=0)

    plane_pts = []
    for r in rs:
        x, y = ring_xy(r)
        z = np.full_like(x, z_plane)
        plane_pts.append(np.stack([x, y, z], axis=-1))
    plane_pts = np.stack(plane_pts, axis=0)

    tris = []
    def add_quad(v00, v10, v11, v01, outward_ccw=True):
        if outward_ccw:
            tris.append(np.array([v00, v10, v11], dtype=float))
            tris.append(np.array([v00, v11, v01], dtype=float))
        else:
            tris.append(np.array([v00, v11, v10], dtype=float))
            tris.append(np.array([v00, v01, v11], dtype=float))

    # Spherical cap (outward ~ +Z if R>0; winding below is correct for outward normals)
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

# =========================
# Sources
# =========================
def point_source_single(origin: np.ndarray, theta_deg: float, phi_deg: float, wavelength_nm: float = 550.0) -> Ray:
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    d = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ], dtype=float)
    return Ray(origin=np.asarray(origin, dtype=float), direction=normalize(d), wavelength_nm=wavelength_nm)

def parallel_beam_single(point_on_plane: np.ndarray, direction: np.ndarray, wavelength_nm: float = 550.0) -> Ray:
    return Ray(origin=np.asarray(point_on_plane, dtype=float),
               direction=normalize(np.asarray(direction, dtype=float)),
               wavelength_nm=wavelength_nm)

def _orthonormal_frame_from_axis(axis: np.ndarray):
    """Return unit vectors (u, v, w) with w || axis."""
    w = normalize(np.asarray(axis, dtype=float))
    # pick a vector not parallel to w
    a = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = normalize(np.cross(a, w))
    v = normalize(np.cross(w, u))
    return u, v, w

def _gaussian_xy(n: int, w_at_plane: float, rng: np.random.Generator):
    """
    Sample (x,y) from 2D Gaussian with I(r) ~ exp(-2 r^2 / w^2).
    That’s equivalent to x,y ~ N(0, sigma^2) with sigma = w / sqrt(2).
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
    Emit a bundle of rays that sample a fundamental Gaussian at the plane z=launch_z.
    Directions follow the local Gaussian wavefront curvature (sphere of radius R(z)).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Gaussian beam parameters
    w0 = float(waist_radius)
    zR = np.pi * w0 * w0 / float(wavelength)           # Rayleigh range
    z = float(launch_z) - float(waist_z)               # distance from waist to plane (sign matters)
    w_z = w0 * np.sqrt(1.0 + (z / zR) ** 2)            # beam radius at plane

    # Wavefront radius R(z) (positive for forward +z propagation)
    if abs(z) < 1e-15:
        Rz = np.inf
    else:
        Rz = z * (1.0 + (zR / z) ** 2)

    # Build frame and center point at the plane
    u, v, w = _orthonormal_frame_from_axis(axis)
    origin_plane = center_xy[0] * u + center_xy[1] * v + (launch_z * w)

    # Sample transverse positions
    xs, ys = _gaussian_xy(n_rays, w_z, rng)

    rays: List[Ray] = []
    # Sphere center for the wavefront the rays must point to:
    # if R=∞, directions are parallel to +w. If finite, aim at plane point + R * w.
    wavefront_center = origin_plane + (Rz * w) if np.isfinite(Rz) else None

    for x, y in zip(xs, ys):
        p = origin_plane + x * u + y * v
        if wavefront_center is None:
            d = w.copy()
        else:
            d = normalize(wavefront_center - p)
        rays.append(Ray(origin=p, direction=d))
    return rays

# =========================
# Intensity sampling on z-planes (point-based heat map)
# =========================

def intersect_path_with_z(path: np.ndarray, z0: float) -> Optional[np.ndarray]:
    """Return intersection point of a (N,3) polyline with plane z=z0, or None if no crossing."""
    z = path[:, 2]
    dz = z[1:] - z[:-1]
    # indices where segment crosses z0 (inclusive/exclusive)
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
    """Intersect many polylines with z=z0. Returns (M,2) xy points for all rays that cross."""
    pts = []
    for path in paths:
        p = intersect_path_with_z(path, z0)
        if p is not None:
            pts.append([p[0], p[1]])
    return np.asarray(pts, dtype=float) if pts else np.empty((0, 2), dtype=float)


# =========================
# Separate scatter-plot intensity at z-planes (+ optional Gaussian fit)
# =========================

def _fit_gaussian_from_points(xy: np.ndarray):
    """
    Estimate 2D Gaussian parameters from samples xy ~ N(mu, Sigma).
    Returns dict with keys: mu (2,), Sigma (2x2), eigvals (2,), eigvecs (2x2),
    sigmas (sqrt eigvals), w (1/e^2 radii = sqrt(2)*sigmas), angle_deg (major-axis rotation).
    """
    mu = xy.mean(axis=0)
    X = xy - mu
    # sample covariance (MLE: 1/N, unbiased: 1/(N-1)); either is fine here
    Sigma = (X.T @ X) / max(1, xy.shape[0]-1)
    # Eigen-decomposition; sort descending by eigenvalue
    evals, evecs = np.linalg.eigh(Sigma)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    sigmas = np.sqrt(np.maximum(evals, 0.0))
    w = np.sqrt(2.0) * sigmas  # 1/e^2 radii
    # Angle of major axis w.r.t +x
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


def _ellipse_points(mu, axes_w, angle_deg, n=200):
    """Return ellipse points for the 1/e^2 contour centered at mu with radii axes_w and rotation angle."""
    t = np.linspace(0, 2*np.pi, n)
    ca, sa = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    R = np.array([[ca, -sa], [sa, ca]])
    pts = (R @ (np.vstack((axes_w[0]*np.cos(t), axes_w[1]*np.sin(t)))))
    pts = pts.T + mu
    return pts


def plot_intensity_scatter(xy: np.ndarray, z0: float, fit: bool = True, bins: int = 0, title_prefix: str = ""):
    """Make a separate 2D plot of ray cross-section at z=z0.
    - xy: (M,2) intersection points
    - fit: whether to estimate and draw 2D Gaussian 1/e^2 ellipse and report w_x, w_y
    - bins: if >0, also show a faint 2D histogram as background
    """
    if xy.size == 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_title(f"{title_prefix} z={z0:.4g} m — no intersections")
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        ax.set_aspect('equal', adjustable='box')
        plt.show()
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if bins and bins > 0:
        # light background heatmap
        H, xedges, yedges = np.histogram2d(xy[:,0], xy[:,1], bins=bins)
        ax.imshow(H.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                  cmap='Greys', alpha=0.35, aspect='equal', interpolation='nearest')

    ax.scatter(xy[:,0], xy[:,1], s=6, alpha=0.7)

    txt = ""
    if fit:
        pars = _fit_gaussian_from_points(xy)
        mu = pars['mu']; w = pars['w']; ang = pars['angle_deg']
        # 1/e^2 ellipse (I / I0 = e^-2)
        ell = _ellipse_points(mu, w, ang)
        ax.plot(ell[:,0], ell[:,1], lw=2)
        txt = f"μ=({mu[0]:.3g},{mu[1]:.3g})  w=(w_x={w[0]*1e3:.2f} mm, w_y={w[1]*1e3:.2f} mm)  angle={ang:.1f}°"

    ax.set_title(f"{title_prefix} z={z0:.4g} m  {txt}")
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def make_intensity_grid_xy(xy: np.ndarray, z0: float,
                           bins: int = 200,
                           margin: float = 1.2,
                           extent_xy: Optional[Tuple[Tuple[float,float], Tuple[float,float]]] = None,
                           cmap_name: str = 'inferno') -> Tuple[pv.ImageData, str]:
    """
    Build a 2D histogram on the plane z=z0 and return a thin ImageData carrying cell-data 'I'.
    - xy: (M,2) sample points
    - bins: number of bins along each axis (int or (nx, ny))
    - margin: scale on auto extents
    - extent_xy: ((xmin,xmax),(ymin,ymax)) override
    Returns (grid, scalar_name)
    """
    # Helper to create an empty thin image so downstream plotting doesn't crash
    def _empty_grid():
        grid = pv.ImageData()
        grid.dimensions = (2, 2, 2)  # points (=> 1x1x1 cells)
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

    H, x_edges, y_edges = np.histogram2d(xy[:,0], xy[:,1], bins=[nx, ny], range=[[xmin, xmax], [ymin, ymax]])

    # Build a thin 3D image so we can store cell data
    grid = pv.ImageData()
    # ImageData dimensions are number of points; to get (nx,ny,1) cells, set dims=(nx+1, ny+1, 2)
    grid.dimensions = (nx + 1, ny + 1, 2)
    dx = (x_edges[-1] - x_edges[0]) / nx
    dy = (y_edges[-1] - y_edges[0]) / ny
    grid.origin = (x_edges[0], y_edges[0], z0 - 1e-6)
    grid.spacing = (dx, dy, 2e-6)

    # VTK expects Fortran-order flatten for cell arrays
    grid.cell_data['I'] = H.T.ravel(order='F')
    grid.cell_data.set_active('I')
    return grid, 'I'

# =========================
# PyVista helpers
# =========================
def tris_to_polydata(tris: np.ndarray) -> pv.PolyData:
    """
    Convert (N,3,3) triangle vertices to a VTK PolyData with shared vertices.
    Deduplicates vertices to keep the mesh light for rendering.
    """
    # Deduplicate points using a view on structured array for stable hashing
    pts = tris.reshape(-1, 3)
    # round to mitigate tiny float diffs from repeated construction (optional)
    rounded = np.round(pts, decimals=12)
    # create mapping
    uniq, inv = np.unique(rounded, axis=0, return_inverse=True)
    faces_idx = inv.reshape(-1, 3)
    # VTK "faces" format: [3, i, j, k, 3, i, j, k, ...]
    faces = np.hstack([np.concatenate(([3], tri)) for tri in faces_idx]).astype(np.int64)
    mesh = pv.PolyData(uniq, faces)
    mesh.clean(inplace=True)
    return mesh

def add_ray_polyline(plotter: pv.Plotter, path_pts: np.ndarray, line_width: float = 3.0, color="red"):
    line = pv.lines_from_points(path_pts, close=False)
    plotter.add_mesh(line, color=color, line_width=line_width)

# Overlay intensity map grid on the plotter (supports ImageData / any DataSet with scalars)
def add_intensity_plane(plotter: pv.Plotter, grid: pv.DataSet, scalar_name: str = 'I', cmap: str = 'inferno', opacity: float = 0.85):
    plotter.add_mesh(grid, scalars=scalar_name, cmap=cmap, opacity=opacity, lighting=False, show_edges=False)

# =========================
# Demo
# =========================
def main(mesh_kind: str = "sphere"):
    # Choose mesh
    if mesh_kind.lower() == "lens":
        tris = plano_convex_tris(
            aperture_radius=100e-3,    # 12.5 mm
            R=160e-3,                    # 50 mm ROC, convex toward +Z
            center_thickness=50e-3,      # 4 mm CT
            radial_segments=64,
            azimuth_segments=256,
            center=(0, 0, 0),
        )
        n_inside = 1.52
        name = "plano_convex"

        # --- Gaussian beam parameters ---
        n_rays      = 100               # how many rays to launch
        lam         = 780e-9            # wavelength [m]
        w0          = 40e-3               # waist radius [m]
        waist_z     = -0.10              # waist position along +Z [m]
        launch_z    = -0.06              # plane where we emit rays [m] (before the lens)
        axis        = np.array([0.0, 0.0, 1.0])  # propagate along +Z

        rays = gaussian_beam_rays(
            n_rays=n_rays,
            waist_radius=w0,
            wavelength=lam,
            waist_z=waist_z,
            launch_z=launch_z,
            axis=axis,
            center_xy=np.array([0.0, 0.0]),  # center on axis; shift to decenter the beam
        )
        max_length = 0.35
    else:
        # default: sphere
        tris = make_icosphere_tris(radius=0.5, levels=3, center=(0, 0, 0))
        n_inside = 1.52
        name = "icosphere"
        ray = parallel_beam_single(point_on_plane=np.array([-1.5, 0.12, 0.08]), direction=np.array([1.0, 0.0, 0.0]))
        max_length = 6.0

    # Build tracer mesh & scene
    mesh_obj = Mesh.from_triangle_array(tris, n_inside=n_inside, name=name)
    scene = Scene([mesh_obj], n_outside=1.0)
    scene.build_accel()

    # ---- Trace bundle (or single ray, if you kept that path) ----
    if 'rays' in locals():
        paths = trace_ray_bundle_parallel(scene, rays, max_bounces=100, max_path_length=max_length)
    else:
        path = trace_single_ray(scene, ray, max_bounces=100, max_path_length=max_length)
        paths = [np.asarray(path)]

    # Optional: separate matplotlib scatter plots at z-planes
    try:
        from __main__ import args as _cli_args
        want_scatter = getattr(_cli_args, 'scatter', False)
        fit_on = not getattr(_cli_args, 'no_fit', False)
        zplanes_cli = getattr(_cli_args, 'zplane', [])
    except Exception:
        want_scatter = False
        fit_on = True
        zplanes_cli = []

    if want_scatter and len(zplanes_cli) > 0:
        for z0 in zplanes_cli:
            xy = intersect_paths_with_z(paths, z0)
            plot_intensity_scatter(xy, z0, fit=fit_on, bins=0, title_prefix=name)

    # Convert to PolyData for rendering
    poly = tris_to_polydata(tris)

    # Render
    p = pv.Plotter(window_size=(900, 700))
    # --- Optional: cross-sectional intensity maps at user-specified z-planes ---
    try:
        from __main__ import args as _cli_args  # reuse parsed args if running as script
        zplanes = getattr(_cli_args, 'zplane', [])
        bins = getattr(_cli_args, 'bins', 200)
    except Exception:
        zplanes = []
        bins = 200

    title = f"{name} (n={n_inside}) — Gaussian ray bundle"
    if len(zplanes) > 0:
        title += f"  |  {len(zplanes)} intensity plane(s)"
    p.add_title(title, font_size=12)
    p.add_mesh(poly, color="lightblue", opacity=0.35, show_edges=True, lighting=True, smooth_shading=True)

    # Add all ray polylines
    for i, pts in enumerate(paths):
        line = pv.lines_from_points(pts, close=False)
        p.add_mesh(line, color="crimson", line_width=2.0, opacity=0.9)

    # --- Optional: overlay intensity heatmaps at z-planes ---
    for z0 in zplanes:
        xy = intersect_paths_with_z(paths, z0)
        grid, sname = make_intensity_grid_xy(xy, z0, bins=bins)
        add_intensity_plane(p, grid, scalar_name=sname, cmap='inferno', opacity=0.8)

    # Optional: mark the launch plane center and a few points
    if 'rays' in locals():
        p.add_mesh(pv.Sphere(radius=0.004, center=rays[0].origin), color="crimson", opacity=0.8)

    p.add_axes(interactive=True)
    p.enable_parallel_projection()  # orthographic viewing; comment out for perspective
    p.show_grid()
    p.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PyVista ray-trace visualization")
    parser.add_argument("--mesh", choices=["sphere", "lens"], default="lens", help="Which mesh to visualize")
    parser.add_argument("--zplane", type=float, nargs="*", default=[0.2], help="One or more z values at which to compute intensity maps / scatter plots")
    parser.add_argument("--bins", type=int, default=200, help="Histogram bins per axis for intensity map overlay (PyVista, if used)")
    parser.add_argument("--scatter", action="store_true", default=True, help="Show a separate matplotlib scatter plot at each z-plane")
    parser.add_argument("--no-fit", action="store_true", default=False, help="Disable Gaussian fit/ellipse on the scatter plot")
    args = parser.parse_args()
    main(mesh_kind=args.mesh)