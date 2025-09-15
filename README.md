This is a quick proof-of-concept for a raytracing engine for optical simulations based on meshed objects. The core functionality involves propagating one or more Rays (`ray_sources.py`) through a Scene (`ray_tracer.py`) made up of Meshes (`geometry.py`). When a collision between a ray and a mesh surface occurs, the ray is either specularly reflected or refracted according to Snell's law, and the propagation is adjusted accordingly.

The structure of the code is as follows:
- `geometry.py`: Defines the basic geometric primitives (Mesh, Triangle) and some utility functions.
- `ray_sources.py`: Defines various ray sources, currently only Gaussian beams.
- `ray_tracer.py`: Implements the ray tracing engine.
- `analysis.py`: Includes analysis utilities for ray tracing results, such as intensity analysis and beam parameter fitting.
- `visualization.py`: Includes 3D visualization utilities for ray tracing results, such as plotting beam cross-sections and evolution.
- `math_utils.py`: Includes some utility functions for linear algebra and geometry.
- `demo.py`: A simple demo script that puts it all together.