from sklearn.decomposition import PCA
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline

def make_circle(coords, radius):
    # Fit PCA
    pca = PCA(n_components=3)
    pca.fit(coords)
    plane_origin = coords.mean(axis=0)
    plane_normal = pca.components_[-1]  # normal vector to the best-fit plane

    # Create circle points
    circle_points = []
    for angle in np.linspace(0, 2 * np.pi, 100):
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0
        circle_points.append(plane_origin + x * pca.components_[0] + y * pca.components_[1] + z * plane_normal)
    return np.array(circle_points)

def get_top_bottom_wall_centres(mesh, wall_coords):
    ## Separate these into top and bottom walls
    # Use KMeans clustering to separate into two groups (top and bottom walls)
    if len(wall_coords) >= 2:
        kmeans = KMeans(n_clusters=2, random_state=0).fit(wall_coords)
        labels = kmeans.labels_
        group1 = wall_coords[labels == 0]
        group2 = wall_coords[labels == 1]
        # Assign top/bottom by comparing mean y values
        if group1[:, 1].mean() > group2[:, 1].mean():
            top_wall_coords = group1
            bottom_wall_coords = group2
        else:
            top_wall_coords = group2
            bottom_wall_coords = group1
    else:
        # Fallback: use all as top, none as bottom
        top_wall_coords = wall_coords
        bottom_wall_coords = np.empty((0, 3))

    # Create the wall traces
    top_wall_trace = go.Scatter3d(
        x=top_wall_coords[:, 0],
        y=top_wall_coords[:, 1],
        z=top_wall_coords[:, 2],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Top Wall Vertices'
    )
    bottom_wall_trace = go.Scatter3d(
        x=bottom_wall_coords[:, 0],
        y=bottom_wall_coords[:, 1],
        z=bottom_wall_coords[:, 2],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Bottom Wall Vertices'
    )

    centre_top = top_wall_coords.mean(axis=0)
    centre_bottom = bottom_wall_coords.mean(axis=0)

    centre_top_trace = go.Scatter3d(
        x=[centre_top[0]],
        y=[centre_top[1]],
        z=[centre_top[2]],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Top Wall Centre'
    )
    centre_bottom_trace = go.Scatter3d(
        x=[centre_bottom[0]],
        y=[centre_bottom[1]],
        z=[centre_bottom[2]],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Bottom Wall Centre'
    )

    return centre_top, centre_bottom, top_wall_coords, bottom_wall_coords, top_wall_trace, bottom_wall_trace, centre_top_trace, centre_bottom_trace

def get_midpoint_cross_section(mesh, centre_top, centre_bottom):
    """
    Take a cross section at the midpoint between two wall centres.
    Returns the cross section points and the midpoint.
    """
    import numpy as np

    wall_vec = centre_top - centre_bottom
    wall_vec /= np.linalg.norm(wall_vec)
    midpoint = (centre_top + centre_bottom) / 2

    section = mesh.section(plane_origin=midpoint, plane_normal=wall_vec)
    if section is not None:
        if hasattr(section, 'discrete'):
            section_points = np.vstack([seg for seg in section.discrete])
        else:
            section_points = section.vertices
    else:
        section_points = None

    return midpoint, section_points

def cross_section_area_2d(points):
        pca = PCA(n_components=2)
        pts2 = pca.fit_transform(points)
        hull = ConvexHull(pts2)
        return hull.volume

def get_centreline_estimate(top_circle_centre, bottom_circle_centre, left_section_centre, right_section_centre):
    points = np.vstack([top_circle_centre, left_section_centre, bottom_circle_centre, right_section_centre, top_circle_centre])  # repeat first point to close
    t = np.linspace(0, 1, len(points))
    t_fine = np.linspace(0, 1, 200)
    spline_x = CubicSpline(t, points[:,0], bc_type='periodic')(t_fine)
    spline_y = CubicSpline(t, points[:,1], bc_type='periodic')(t_fine)
    spline_z = CubicSpline(t, points[:,2], bc_type='periodic')(t_fine)
    spline_trace = go.Scatter3d(
        x=spline_x, y=spline_y, z=spline_z,
        mode='lines',
        line=dict(color='black', width=6),
        name='Central Spline (Closed)'
        )
    return spline_trace, spline_x, spline_y, spline_z

def extract_open_spline_between_anchors(spline, anchor1, anchor2):
    # Find closest indices to anchors
    idx1 = np.argmin(np.linalg.norm(spline - anchor1, axis=1))
    idx2 = np.argmin(np.linalg.norm(spline - anchor2, axis=1))
    # Extract both possible open paths
    if idx1 < idx2:
        segment1 = spline[idx1:idx2+1]
        segment2 = np.vstack([spline[idx2:], spline[:idx1+1]])
    else:
        segment1 = spline[idx2:idx1+1]
        segment2 = np.vstack([spline[idx1:], spline[:idx2+1]])
    return segment1, segment2

def get_regularly_spaced_cross_sections(mesh, smoothed, centre_top, centre_bottom, num_sections=30):
    """
    Optimized version: Given a mesh and a smoothed centreline (Nx3 array), return
    cross section points at regularly spaced intervals along the centreline,
    avoiding points too close to the top and bottom wall centroids.
    Uses batched nearest face search for barycentric data.
    """
    import numpy as np
    smoothed_x = smoothed[:,0]
    smoothed_y = smoothed[:,1]
    smoothed_z = smoothed[:,2]
    smoothed_points = np.column_stack([smoothed_x, smoothed_y, smoothed_z])

    dists = np.linalg.norm(np.diff(smoothed_points, axis=0), axis=1)
    arc_length = np.concatenate([[0], np.cumsum(dists)])
    total_length = arc_length[-1]
    target_lengths = np.linspace(0, total_length, num_sections + 4)

    # Interpolate to get regularly spaced points
    interp_points = np.empty((len(target_lengths), 3))
    for i in range(3):
        interp_points[:, i] = np.interp(target_lengths, arc_length, smoothed_points[:, i])

    smoothed_points = interp_points

    # Find indices of closest points to top and bottom wall centroids
    top_idx = np.argmin(np.linalg.norm(smoothed_points - centre_top, axis=1))
    bottom_idx = np.argmin(np.linalg.norm(smoothed_points - centre_bottom, axis=1))

    # Sample num_sections + 4 indices evenly along the centreline (to allow for ±1 removal at each wall)
    indices = np.linspace(0, len(smoothed_points) - 1, num_sections + 4, dtype=int)

    def remove_near_wall(idx, indices, window=1):
        closest = np.argmin(np.abs(indices - idx))
        to_remove = [(closest + offset) % len(indices) for offset in range(-window, window+1)]
        return set(to_remove)

    remove_set = remove_near_wall(top_idx, indices, window=1) | remove_near_wall(bottom_idx, indices, window=1)

    keep_mask = np.ones(len(indices), dtype=bool)
    for i in remove_set:
        keep_mask[i] = False

    final_indices = indices[keep_mask]

    section_points_list = []
    section_traces = []
    section_bary_data = []

    # Extract cross-sections at the selected indices
    for idx in final_indices:
        midpoint = smoothed_points[idx]
        
        # Compute tangent vector (simplified version)
        if idx == 0:
            tangent = smoothed_points[idx+1] - smoothed_points[idx]
        elif idx == len(smoothed_points) - 1:
            tangent = smoothed_points[idx] - smoothed_points[idx-1]
        else:
            tangent = smoothed_points[idx+1] - smoothed_points[idx-1]
        
        tangent = tangent / np.linalg.norm(tangent)
        
        # Extract cross-section
        try:
            section = mesh.section(plane_origin=midpoint, plane_normal=tangent)
            if section is not None:
                if hasattr(section, 'discrete'):
                    section_points = np.vstack([seg for seg in section.discrete])
                else:
                    section_points = section.vertices
                
                if section_points is not None and len(section_points) > 0:
                    section_points_list.append(section_points)
                    section_trace = go.Scatter3d(
                        x=section_points[:, 0],
                        y=section_points[:, 1],
                        z=section_points[:, 2],
                        mode='markers',
                        marker=dict(size=5, color='green'),
                        name=f'Section {idx}'
                    )
                    section_traces.append(section_trace)
                    section_bary_data.append([])  # Simplified - no barycentric data
        except Exception as e:
            print(f"Warning: Section extraction failed at index {idx}: {e}")
            continue

    return section_points_list, section_traces, section_bary_data