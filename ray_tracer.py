#!/usr/bin/env python3
"""
Ray tracing engine for optical simulations.
Handles ray-surface intersections and propagation through optical media.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from geometry import Mesh, Triangle
from ray_sources import Ray
from math_utils import (
    normalize, reflect_nb, refract_one_sided_nb, reflect, refract_one_sided,
    mt_closest_intersection, EPS
)
import numba as nb

@dataclass
class Hit:
    """Ray-surface intersection result."""
    t: float
    point: np.ndarray
    normal_out: np.ndarray
    mesh: Mesh
    tri_index: int

class Scene:
    """A scene containing multiple optical meshes."""
    
    def __init__(self, meshes: List[Mesh], n_outside: float = 1.0):
        self.meshes = meshes
        self.n_outside = n_outside
        # Acceleration structure for fast intersection
        self._V0 = None
        self._E1 = None
        self._E2 = None
        self._Nout = None
        self._tri_mesh_idx = None

    def build_accel(self):
        """Build acceleration structure for fast ray-triangle intersection."""
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
        """Find closest ray-surface intersection."""
        # Fast path: Numba-accelerated array traversal
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

def trace_single_ray(scene: Scene,
                     ray: Ray,
                     max_bounces: int = 100,
                     max_path_length: float = 1e6) -> List[np.ndarray]:
    """
    Trace a single ray through the scene.
    
    Returns list of 3D points representing the ray path.
    """
    pts = [ray.origin.copy()]
    o = ray.origin.copy()
    d = normalize(ray.direction.copy())

    # Medium stack: (mesh_or_None, refractive_index)
    med_stack: List[Tuple[Optional[Mesh], float]] = [(None, scene.n_outside)]

    path_len = 0.0
    bounces = 0
    
    while bounces < max_bounces and path_len < max_path_length:
        hit = scene.closest_intersection(o, d)
        if hit is None:
            # Ray escapes to infinity
            pts.append(o + d * (min(200.0, max_path_length - path_len)))
            break

        p = hit.point
        path_len += np.linalg.norm(p - o)
        pts.append(p.copy())

        # Determine current and destination media
        top_mesh, n1 = med_stack[-1]
        entering = (top_mesh is not hit.mesh)
        n2 = hit.mesh.n_inside if entering else (med_stack[-2][1] if len(med_stack) > 1 else scene.n_outside)

        # Orient normal to point into destination medium
        N_to_n2 = hit.normal_out if entering else (-hit.normal_out)

        # Apply Snell's law or reflection
        if nb is not None:
            T, tir = refract_one_sided_nb(d, N_to_n2, n1, n2)
            if tir:
                d = reflect_nb(d, hit.normal_out)
            else:
                d = T
                if entering:
                    med_stack.append((hit.mesh, hit.mesh.n_inside))
                else:
                    if len(med_stack) > 1:
                        med_stack.pop()
        else:
            T, tir = refract_one_sided(d, N_to_n2, n1, n2)
            if tir:
                d = reflect(d, hit.normal_out)
            else:
                d = T
                if entering:
                    med_stack.append((hit.mesh, hit.mesh.n_inside))
                else:
                    if len(med_stack) > 1:
                        med_stack.pop()

        # Move slightly off surface to avoid self-intersection
        o = p + d * (10 * EPS)
        bounces += 1

    return pts

def trace_ray_bundle(scene: Scene, rays: List[Ray],
                     max_bounces: int = 100, max_path_length: float = 1e6) -> List[np.ndarray]:
    """Trace multiple rays sequentially."""
    paths = []
    for r in rays:
        pts = trace_single_ray(scene, r, max_bounces=max_bounces, max_path_length=max_path_length)
        paths.append(np.asarray(pts))
    return paths

def trace_ray_bundle_parallel(scene: Scene, rays: List[Ray], 
                              max_bounces: int = 100, max_path_length: float = 1e6, 
                              workers: Optional[int] = None) -> List[np.ndarray]:
    """Trace multiple rays in parallel for better performance."""
    paths = [None] * len(rays)
    
    # Ensure acceleration structure is built
    if hasattr(scene, 'build_accel'):
        scene.build_accel()
    
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(trace_single_ray, scene, r, max_bounces, max_path_length): i 
                for i, r in enumerate(rays)}
        for f in as_completed(futs):
            i = futs[f]
            pts = f.result()
            paths[i] = np.asarray(pts)
    
    return paths
