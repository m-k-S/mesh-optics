#!/usr/bin/env python3
"""
Demonstration script for the cleaned-up optical ray tracing system.
Shows lens ray tracing with Gaussian beam sources and analysis capabilities.
"""

import numpy as np
import argparse
from typing import List

from geometry import Mesh, plano_convex_tris
from ray_sources import gaussian_beam_rays
from ray_tracer import Scene, trace_ray_bundle_parallel
from analysis import intersect_paths_with_z, plot_beam_cross_section, analyze_beam_evolution, plot_beam_evolution
from visualization import tris_to_polydata, create_optical_scene, make_intensity_grid_xy

def create_demo_lens(aperture_radius: float = 100e-3, 
                     R: float = 160e-3, 
                     center_thickness: float = 50e-3,
                     n_inside: float = 1.52) -> Mesh:
    """Create a demonstration plano-convex lens."""
    tris = plano_convex_tris(
        aperture_radius=aperture_radius,
        R=R,
        center_thickness=center_thickness,
        radial_segments=64,
        azimuth_segments=256,
        center=(0, 0, 0),
    )
    return Mesh.from_triangle_array(tris, n_inside=n_inside, name="plano_convex_lens")

def create_demo_gaussian_beam(n_rays: int = 100,
                              waist_radius: float = 40e-3,
                              wavelength: float = 780e-9,
                              waist_z: float = -0.10,
                              launch_z: float = -0.06):
    """Create a demonstration Gaussian beam."""
    return gaussian_beam_rays(
        n_rays=n_rays,
        waist_radius=waist_radius,
        wavelength=wavelength,
        waist_z=waist_z,
        launch_z=launch_z,
        axis=np.array([0.0, 0.0, 1.0]),
        center_xy=np.array([0.0, 0.0]),
    )

def main():
    parser = argparse.ArgumentParser(description="Optical ray tracing demonstration")
    parser.add_argument("--n-rays", type=int, default=100, help="Number of rays to trace")
    parser.add_argument("--zplane", type=float, nargs="*", default=[0.2], 
                       help="Z-positions for intensity analysis")
    parser.add_argument("--bins", type=int, default=200, 
                       help="Histogram bins for intensity maps")
    parser.add_argument("--scatter", action="store_true", default=True,
                       help="Show separate matplotlib scatter plots")
    parser.add_argument("--no-fit", action="store_true", default=False,
                       help="Disable Gaussian fitting on scatter plots")
    parser.add_argument("--evolution", action="store_true", default=False,
                       help="Show beam evolution analysis")
    args = parser.parse_args()

    print("Creating optical system...")
    
    # Create lens and scene
    lens_mesh = create_demo_lens()
    scene = Scene([lens_mesh], n_outside=1.0)
    scene.build_accel()
    
    # Create Gaussian beam
    rays = create_demo_gaussian_beam(n_rays=args.n_rays)
    
    print(f"Tracing {len(rays)} rays...")
    
    # Trace rays
    paths = trace_ray_bundle_parallel(scene, rays, max_bounces=100, max_path_length=0.35)
    
    print("Creating visualizations...")
    
    # Convert mesh for visualization
    tris = plano_convex_tris(100e-3, 160e-3, 50e-3, 64, 256, (0, 0, 0))
    poly = tris_to_polydata(tris)
    
    # Prepare intensity planes for 3D visualization
    intensity_planes = []
    for z0 in args.zplane:
        xy = intersect_paths_with_z(paths, z0)
        if len(xy) > 0:
            grid, sname = make_intensity_grid_xy(xy, z0, bins=args.bins)
            intensity_planes.append((grid, sname))
    
    # Create 3D visualization
    title = f"Plano-Convex Lens (n=1.52) — {len(rays)} Gaussian rays"
    if len(args.zplane) > 0:
        title += f"  |  {len(args.zplane)} intensity plane(s)"
    
    plotter = create_optical_scene(
        mesh_polydata=poly,
        ray_paths=paths,
        title=title,
        intensity_planes=intensity_planes
    )
    
    # Add source point marker
    if rays:
        import pyvista as pv
        plotter.add_mesh(pv.Sphere(radius=0.004, center=rays[0].origin), 
                        color="crimson", opacity=0.8)
    
    # Show 3D visualization
    plotter.show()
    
    # Optional: separate matplotlib scatter plots
    if args.scatter and len(args.zplane) > 0:
        print("Creating cross-section plots...")
        fit_on = not args.no_fit
        for z0 in args.zplane:
            xy = intersect_paths_with_z(paths, z0)
            plot_beam_cross_section(xy, z0, fit=fit_on, bins=0, 
                                   title_prefix="Plano-Convex Lens")
    
    # Optional: beam evolution analysis
    if args.evolution:
        print("Analyzing beam evolution...")
        z_positions = np.linspace(-0.05, 0.3, 50)
        results = analyze_beam_evolution(paths, z_positions)
        plot_beam_evolution(results, "Plano-Convex Lens")
    
    print("Demo completed!")

if __name__ == "__main__":
    main()
