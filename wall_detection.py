from scipy.spatial import cKDTree
import numpy as np

def get_wall_vertices(gc1, gc2):
  
    # KDTree lets us match within a small tolerance
    tree = cKDTree(gc1.vertices)
    dist, idx = tree.query(gc2.vertices, distance_upper_bound=1e-5)

    # Vertices of outer that are close to inner
    shared_gc2_idx = np.where(np.isfinite(dist))[0]
    shared_gc1_idx = idx[shared_gc2_idx]

    return shared_gc2_idx, shared_gc1_idx