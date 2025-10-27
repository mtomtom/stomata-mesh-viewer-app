import numpy as np
import trimesh
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline

## Get the areas of the midsections
def cross_section_area_2d(points):
        pca = PCA(n_components=2)
        pts2 = pca.fit_transform(points)
        hull = ConvexHull(pts2)
        return hull.volume

def get_circle_trace(circle, name="Circle", colour="red"):
    ## Create the circle traces
    circle_trace = go.Scatter3d(
        x=circle[:, 0],
        y=circle[:, 1],
        z=circle[:, 2],
        mode='lines',
        line=dict(color=colour, width=2),
        name=name
    )
    return circle_trace

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

def get_barycentric_coords(point, face_vertices):
    # face_vertices: (3, 3) array
    v0 = face_vertices[1] - face_vertices[0]
    v1 = face_vertices[2] - face_vertices[0]
    v2 = point - face_vertices[0]
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w])

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

def calculate_cross_section_aspect_ratios(sections_points_list):
    """Compute aspect ratio (major/minor) for each cross section.

    Parameters
    ----------
    sections_points_list : list[array_like] or ndarray
        Either a list where each element is an (N_i, 3) array of points for a
        cross section, OR a single (N, 3) array (treated as one section).

    Returns
    -------
    list[float]
        Aspect ratio per cross section. 0.0 for invalid/degenerate sections.

    Notes
    -----
    1. A PCA is fit per section to obtain a local 2D plane.
    2. The *major* axis for each section is chosen as the PCA axis most
       aligned (by absolute dot) to the midpoint section's first PCA axis.
    3. The *minor* axis is enforced as the perpendicular in the 2D plane.
    4. Width/height are taken as peak-to-peak extents along major/minor.
    """
    from sklearn.decomposition import PCA
    import numpy as np

    # Normalize input: allow a single (N,3) array
    if isinstance(sections_points_list, np.ndarray):
        if sections_points_list.ndim == 2 and sections_points_list.shape[1] == 3:
            sections_points_list = [sections_points_list]
        else:
            raise ValueError("If passing a numpy array it must be shape (N,3).")

    if not sections_points_list:
        return []

    # Ensure all valid sections have >=3 pts; collect indices of valid sections
    valid_indices = [i for i, s in enumerate(sections_points_list) if s is not None and len(s) >= 3]
    if not valid_indices:
        return [0.0] * len(sections_points_list)

    # Midpoint index chosen among the *list* (not only valid subset) for stability
    mid_idx = len(sections_points_list) // 2
    if sections_points_list[mid_idx] is None or len(sections_points_list[mid_idx]) < 3:
        # Fallback: pick the central valid index
        mid_idx = valid_indices[len(valid_indices)//2]

    mid_points = np.asarray(sections_points_list[mid_idx])
    pca_mid = PCA(n_components=2)
    pca_mid.fit(mid_points)
    major_axis_ref = pca_mid.components_[0]

    aspect_ratios = []
    for section in sections_points_list:
        if section is None or len(section) < 3:
            aspect_ratios.append(0.0)
            continue
        pts = np.asarray(section)
        if pts.ndim != 2 or pts.shape[1] != 3:
            aspect_ratios.append(0.0)
            continue
        # Fit PCA (2D) on the 3D points
        pca = PCA(n_components=2)
        section_2d = pca.fit_transform(pts)  # shape (N,2) in PCA basis
        comps = pca.components_              # shape (2,3) in 3D
        # Decide which PCA component is the major axis (aligned with reference)
        dot0 = abs(np.dot(comps[0], major_axis_ref))
        dot1 = abs(np.dot(comps[1], major_axis_ref))
        if dot0 >= dot1:
            width_vals = section_2d[:, 0]
            height_vals = section_2d[:, 1]
        else:
            # swap if second component is closer to reference
            width_vals = section_2d[:, 1]
            height_vals = section_2d[:, 0]
        width = width_vals.max() - width_vals.min()
        height = height_vals.max() - height_vals.min()
        if width <= 1e-12 or height <= 1e-12:
            aspect_ratios.append(0.0)
        else:
            aspect_ratios.append(max(width, height) / min(width, height))
    return aspect_ratios

def visualize_mesh(mesh, extra_details=None, title="Mesh Visualization"):
    traces = [
        go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            color="#0072B2",
            opacity=0.65,
            name='Mesh'
        )
    ]
    if extra_details is not None:
        if isinstance(extra_details, list):
            traces.extend(extra_details)
        else:
            traces.append(extra_details)
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        ),
        title=title
    )
    fig.show()
    return fig

def get_regularly_spaced_cross_sections_batch(mesh, smoothed, centre_top, centre_bottom, num_sections=30):
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

    # Collect all section points for batch nearest search
    all_pts = []
    section_lengths = []
    for idx in final_indices:
        section_points = get_cross_section_points(mesh, smoothed_points, [idx])
        if section_points and section_points[0] is not None:
            section_points = section_points[0]
            section_points_list.append(section_points)
            section_lengths.append(len(section_points))
            section_trace = go.Scatter3d(
                x=section_points[:, 0],
                y=section_points[:, 1],
                z=section_points[:, 2],
                mode='markers',
                marker=dict(size=5, color='green'),
                name=f'Section {idx}'
            )
            section_traces.append(section_trace)
            all_pts.append(section_points)

    if len(all_pts) > 0:
        all_pts_flat = np.vstack(all_pts)
        _, _, face_indices = mesh.nearest.on_surface(all_pts_flat)
        # Split face_indices for each section
        split_indices = np.split(face_indices, np.cumsum(section_lengths)[:-1])
        for section, face_idx_list in zip(section_points_list, split_indices):
            section_bary = []
            for pt, face_idx in zip(section, face_idx_list):
                face_vertices = mesh.vertices[mesh.faces[face_idx]]
                bary = get_barycentric_coords(pt, face_vertices)
                section_bary.append((face_idx, bary))
            section_bary_data.append(section_bary)
    else:
        section_bary_data = [[] for _ in section_points_list]

    return section_points_list, section_traces, section_bary_data

def curve_length(x, y, z):
    # Stack coordinates into (N, 3) array
    points = np.column_stack((x, y, z))
    # Compute distances between consecutive points
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    # Sum to get total length
    return segment_lengths.sum()

def is_closed_loop(points, threshold_ratio=0.05):
    """
    Determine if a 3D point sequence forms a closed loop.
    """
    if points is None or len(points) < 4:
        return False

    points = np.asarray(points)
    centroid = points.mean(axis=0)
    radii = np.linalg.norm(points - centroid, axis=1)
    mean_radius = np.mean(radii)
    if mean_radius == 0:
        return False

    # Check start–end closure
    end_dist = np.linalg.norm(points[0] - points[-1])

    # Check continuity — average gap size between successive points
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    mean_gap = np.mean(diffs)
    max_gap = np.max(diffs)

    # Consider loop closed if start and end join and there are no large internal breaks
    return (end_dist < threshold_ratio * mean_radius) and (max_gap < 2 * mean_gap)


def get_cross_section_points(
    mesh,
    centreline,
    indices,
    wall_coords=None,
    threshold_factor=0.05,
    return_failed=False
):
    """
    Extract cross-section points from a 3D mesh at specific centreline indices.

    Parameters
    ----------
    mesh : pyvista.PolyData or trimesh.Trimesh
        3D mesh to slice.
    centreline : (N, 3) array
        Ordered coordinates of the mesh centreline.
    indices : list[int]
        Indices along the centreline where cross-sections are extracted.
    wall_coords : (M, 3) array, optional
        Additional wall coordinates used to supplement incomplete sections.
    threshold_factor : float, optional
        Fraction of the mesh bounding-box diagonal used as plane thickness (default 0.05).
    return_failed : bool, optional
        If True, returns a list with `None` for failed indices instead of skipping them.

    Returns
    -------
    sections_points_list : list[np.ndarray or None]
        List of cross-section point arrays (each (K, 3)).
    """

    # ---- Compute tangents safely ----
    tangents = np.gradient(centreline, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = np.divide(tangents, np.clip(norms, 1e-8, None))

    # ---- Get global scale ----
    try:
        extents = mesh.bounds[1::2] - mesh.bounds[::2]
        mesh_size = np.linalg.norm(extents)
    except Exception:
        # Fallback if .bounds unavailable
        mesh_size = np.linalg.norm(centreline[-1] - centreline[0])

    threshold = max(threshold_factor * np.linalg.norm(mesh.bounds[1::2] - mesh.bounds[::2]), 0.01)
    threshold = max(threshold, 0.02 * mesh_size)


    sections_points_list = []

    for idx in indices:
        midpoint = centreline[idx]
        tangent = tangents[idx]

        # --- Slice the mesh ---
        try:
            section = mesh.section(plane_origin=midpoint, plane_normal=tangent)
        except Exception as e:
            print(f"[Warning] Section failed at index {idx}: {e}")
            if return_failed:
                sections_points_list.append(None)
            continue

        if section is None:
            if return_failed:
                sections_points_list.append(None)
            continue

        # --- Handle multiple discrete contours ---
        section_points = None
        try:
            if hasattr(section, "discrete") and section.discrete:
                segments = section.discrete
                centroids = [seg.mean(axis=0) for seg in segments]
                dists = [np.linalg.norm(c - midpoint) for c in centroids]
                nearest_seg = segments[int(np.argmin(dists))]
                section_points = nearest_seg
            elif hasattr(section, "vertices"):
                if section.vertices.shape[0] > 0:
                    section_points = order_points_consistently(
                        section.vertices, tangent, midpoint
                    )
        except Exception as e:
            print(f"[Warning] Invalid section geometry at index {idx}: {e}")
            if return_failed:
                sections_points_list.append(None)
            continue

        if section_points is None:
            if return_failed:
                sections_points_list.append(None)
            continue

        # --- Supplement incomplete sections with wall points ---
        if wall_coords is not None:
            plane_normal = tangent / np.linalg.norm(tangent)
            plane_origin = midpoint
            dists_to_plane = np.abs(np.dot(wall_coords - plane_origin, plane_normal))
            wall_pts_near_plane = wall_coords[dists_to_plane < threshold]

            if len(wall_pts_near_plane) > 0:
                projected = (
                    wall_pts_near_plane
                    - np.dot((wall_pts_near_plane - plane_origin), plane_normal)[:, None]
                    * plane_normal
                )
                combined = np.vstack([section_points, projected])
                
                # Filter out duplicate points (within 1% of mesh size)
                from scipy.spatial.distance import cdist
                min_dist = 0.01 * mesh_size
                keep_mask = np.ones(len(combined), dtype=bool)
                for i in range(1, len(combined)):
                    dists = cdist([combined[i]], combined[:i])[0]
                    if np.any(dists < min_dist):
                        keep_mask[i] = False
                combined = combined[keep_mask]
                
                section_points = order_points_consistently(
                    combined, plane_normal, plane_origin
                )

        sections_points_list.append(section_points)

    return sections_points_list


def order_points_consistently(points, normal, midpoint):
    # build a stable 2D basis in the slicing plane
    ref = np.array([0,0,1]) if abs(np.dot(normal,[0,0,1])) < 0.9 else np.array([1,0,0])
    v1 = np.cross(normal, ref); v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)

    R = points - midpoint
    coords2d = np.column_stack([R @ v1, R @ v2])
    angles = np.arctan2(coords2d[:,1], coords2d[:,0])
    order = np.argsort(angles)
    return points[order]

import numpy as np
import trimesh

import numpy as np
import trimesh

import numpy as np
import trimesh

import numpy as np
import trimesh

def find_wall_vertices_vertex_normals(
    mesh: trimesh.Trimesh,
    dot_thresh: float = 0.2,          # kept for API compatibility (unused in this version)
    angle_thresh_deg: float = 150.0,  # crease threshold on dihedral angle (deg), larger = sharper
    align_thresh: float = 0.7,        # |dot(edge_dir, length_axis)| min alignment
    num_length_bins: int = 24,        # lengthwise bins to compute local width centers
    center_percentile: float = 10.0,  # keep edges within this percentile of width-center per bin
    max_components: int = 2           # return up to this many longest seam components (2 for open, 1 for closed)
) -> np.ndarray:
    """
    Robustly identify vertices on the shared wall seam edges.

    Approach
    --------
    1) Compute PCA axes: length (main) and width.
    2) Use dihedral creases (large face adjacency angle) to get candidate edges.
    3) Keep only edges whose direction aligns with the length axis (vertical seam-like).
    4) For each length bin, compute a local width center (mid of min/max); keep only edges
       whose midpoints are in the narrow percentile band around that center.
    5) Build connected components on the kept edges; return vertices from the longest 1–2
       components spanning end-to-end along length.

    Returns
    -------
    np.ndarray
        Unique vertex indices that lie on the seam edge(s).
    """
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    if V.size == 0 or F.size == 0:
        return np.array([], dtype=int)

    # 1) PCA axes
    from sklearn.decomposition import PCA
    mu = V.mean(axis=0)
    comps = PCA(n_components=3).fit(V - mu).components_
    length_axis = comps[0] / (np.linalg.norm(comps[0]) + 1e-12)
    width_axis  = comps[1] / (np.linalg.norm(comps[1]) + 1e-12)

    # 2) Dihedral crease edges
    # Use trimesh adjacency info
    adj_faces = mesh.face_adjacency                 # (M, 2)
    adj_edges = mesh.face_adjacency_edges           # (M, 2) vertex indices of the shared edge
    adj_angles = mesh.face_adjacency_angles         # (M,) radians

    # Large angles (near pi) are sharp creases
    min_angle = np.deg2rad(angle_thresh_deg)
    keep_adj = adj_angles >= min_angle
    if not np.any(keep_adj):
        # fallback: relax threshold
        keep_adj = adj_angles >= np.deg2rad(120.0)

    E = adj_edges[keep_adj]                         # candidate crease edges (K, 2)

    if E.shape[0] == 0:
        return np.array([], dtype=int)

    # Edge vectors and alignment to length axis
    e_vec = V[E[:, 1]] - V[E[:, 0]]
    e_len = np.linalg.norm(e_vec, axis=1, keepdims=True)
    e_dir = np.divide(e_vec, np.clip(e_len, 1e-12, None))
    align = np.abs(e_dir @ length_axis)
    keep_align = align[:, 0] >= align_thresh
    E = E[keep_align]
    if E.shape[0] == 0:
        return np.array([], dtype=int)

    # 3) Center-band filter per length bin
    L = V @ length_axis
    W = V @ width_axis
    # Edge midpoints in projected coords
    M = 0.5 * (V[E[:, 0]] + V[E[:, 1]])
    Lm = M @ length_axis
    Wm = M @ width_axis

    # Quantile edges for length bins
    qs = np.linspace(0.0, 1.0, num_length_bins + 1)
    lbins = np.quantile(L, qs)

    keep_center = np.zeros(E.shape[0], dtype=bool)
    pct = float(center_percentile)
    pct = np.clip(pct, 0.1, 50.0)  # guardrails

    for i in range(num_length_bins):
        lo, hi = lbins[i], lbins[i + 1]
        if i < num_length_bins - 1:
            mask_edges = (Lm >= lo) & (Lm < hi)
            mask_verts = (L >= lo) & (L < hi)
        else:
            mask_edges = (Lm >= lo) & (Lm <= hi)
            mask_verts = (L >= lo) & (L <= hi)

        idx_edges = np.where(mask_edges)[0]
        idx_verts = np.where(mask_verts)[0]
        if idx_edges.size == 0 or idx_verts.size < 3:
            continue

        w_slice = W[idx_verts]
        w_center = 0.5 * (float(w_slice.min()) + float(w_slice.max()))
        d = np.abs(Wm[idx_edges] - w_center)

        # Keep the tight percentile band around center
        thr = np.percentile(d, pct)
        keep_center[idx_edges] = d <= thr

    E = E[keep_center]
    if E.shape[0] == 0:
        return np.array([], dtype=int)

    # 4) Connected components over edges, scored by length span along L
    # Build simple adjacency (vertex -> incident edges)
    from collections import defaultdict, deque

    inc = defaultdict(list)
    for ei, (a, b) in enumerate(E):
        inc[a].append(ei)
        inc[b].append(ei)

    visited_edges = np.zeros(E.shape[0], dtype=bool)
    components = []

    for start_e in range(E.shape[0]):
        if visited_edges[start_e]:
            continue
        # BFS over edges via shared vertices
        q = deque([start_e])
        visited_edges[start_e] = True
        comp_edges = [start_e]
        comp_verts = set([int(E[start_e, 0]), int(E[start_e, 1])])

        while q:
            ei = q.popleft()
            a, b = int(E[ei, 0]), int(E[ei, 1])
            for v in (a, b):
                for ej in inc[v]:
                    if not visited_edges[ej]:
                        visited_edges[ej] = True
                        q.append(ej)
                        comp_edges.append(ej)
                        comp_verts.add(int(E[ej, 0]))
                        comp_verts.add(int(E[ej, 1]))

        comp_verts_list = np.fromiter(comp_verts, dtype=int)
        if comp_verts_list.size == 0:
            continue
        # Score: span along length axis
        span = float(L[comp_verts_list].max() - L[comp_verts_list].min())
        components.append((span, comp_verts_list))

    if not components:
        return np.array([], dtype=int)

    # Keep the longest components (two for open stomata, one for closed)
    components.sort(key=lambda x: x[0], reverse=True)
    chosen = components[:max_components]

    wall_indices = np.unique(np.concatenate([c[1] for c in chosen]))
    return wall_indices

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

def get_midpoint_cross_section_from_centres(mesh, centre_top, centre_bottom):
    """
    Take a cross section at the midpoint between two precomputed wall centres.
    The cross section plane is perpendicular to the line joining the centres.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to section.
    centre_top : np.ndarray
        3D coordinates of the top wall centre.
    centre_bottom : np.ndarray
        3D coordinates of the bottom wall centre.

    Returns
    -------
    midpoint : np.ndarray
        The midpoint between wall centres.
    traces : list
        Plotly traces for visualization.
    section_points : np.ndarray
        Points of the cross section at the midpoint.
    local_axes : np.ndarray
        3x3 array: [wall_vec, left_right_vec, normal_vec]
    """
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA

    # Define wall axis (from bottom to top)
    wall_vec = centre_top - centre_bottom
    wall_vec /= np.linalg.norm(wall_vec)

    # Midpoint
    midpoint = (centre_top + centre_bottom) / 2

    # Find all mesh vertices near the midpoint plane (for local PCA)
    verts = mesh.vertices
    dists = np.dot(verts - midpoint, wall_vec)
    close_mask = np.abs(dists) < np.percentile(np.abs(dists), 10)  # 10% closest to plane
    midplane_points = verts[close_mask]

    # Use PCA to find the two main axes in the midplane
    pca = PCA(n_components=2)
    pca.fit(midplane_points)
    left_right_vec = pca.components_[0]
    normal_vec = np.cross(wall_vec, left_right_vec)
    normal_vec /= np.linalg.norm(normal_vec)

    # Take cross section at midpoint, normal to wall_vec
    section = mesh.section(plane_origin=midpoint, plane_normal=wall_vec)
    if section is not None:
        if hasattr(section, 'discrete'):
            section_points = np.vstack([seg for seg in section.discrete])
        else:
            section_points = section.vertices
    else:
        section_points = None

    # Visualization traces
    midpoint_trace = go.Scatter3d(
        x=[midpoint[0]], y=[midpoint[1]], z=[midpoint[2]],
        mode='markers', marker=dict(size=8, color='black'), name='Midpoint (centres)'
    )
    traces = [midpoint_trace]
    if section_points is not None:
        traces.append(go.Scatter3d(
            x=section_points[:, 0], y=section_points[:, 1], z=section_points[:, 2],
            mode='markers', marker=dict(size=5, color='orange'), name='Midpoint Cross Section (centres)'
        ))

    local_axes = np.stack([wall_vec, left_right_vec, normal_vec], axis=0)

    return midpoint, traces, section_points, local_axes

def split_mesh_at_wall_vertices(mesh, wall_vertices, left_centroid, right_centroid):
    """
    Split mesh into left and right guard cells using the wall vertices,
    without using a slicing plane.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    wall_vertices : (M,) indices of the wall
    left_centroid, right_centroid : (3,) np.arrays
        Approximate centers of left and right guard cells

    Returns
    -------
    left_mesh, right_mesh : trimesh.Trimesh
    """
    vertices = mesh.vertices

    # Define splitting plane at midpoint between centroids
    split_point = (left_centroid + right_centroid) / 2
    split_normal = (right_centroid - left_centroid)
    split_normal /= np.linalg.norm(split_normal)
    
    # Classify vertices by which side of plane they're on
    dists = np.dot(vertices - split_point, split_normal)
    left_mask = dists < 0  # Left side of plane
    right_mask = dists >= 0  # Right side of plane


    # Include wall in both
    left_mask[wall_vertices] = True
    right_mask[wall_vertices] = True

    # Filter faces
    left_faces_mask = np.all(left_mask[mesh.faces], axis=1)
    right_faces_mask = np.all(right_mask[mesh.faces], axis=1)

    left_faces = mesh.faces[left_faces_mask]
    right_faces = mesh.faces[right_faces_mask]

    # Remap vertex indices
    def remap_vertices(mask, faces):
        old_to_new = np.full(len(vertices), -1, dtype=int)
        old_to_new[np.where(mask)[0]] = np.arange(np.sum(mask))
        return trimesh.Trimesh(vertices=vertices[mask], faces=old_to_new[faces], process=True)

    left_mesh = remap_vertices(left_mask, left_faces)
    right_mesh = remap_vertices(right_mask, right_faces)

    return left_mesh, right_mesh

def refine_centre_line_with_three_anchors(
    mesh,
    spline_x,
    spline_y,
    spline_z,
    centre_top,
    centre_bottom,
    side_section_centre,
    num_sections=40
):
    """
    Refine the centreline using only three anchor points (top, bottom, one side),
    passing through those anchors and refined by cross-section centroids.
    The curve will pass through the three anchors and be smoothed by cross-section centroids.
    The result is an open curve (not a loop).
    """
    import numpy as np
    from scipy.interpolate import splprep, splev
    import plotly.graph_objects as go

    # Build cross-section centroids along the initial spline
    spline_points = np.column_stack([spline_x, spline_y, spline_z])
    indices = np.linspace(0, len(spline_points) - 1, num_sections, dtype=int)
    section_points_list = []
    section_centroids = []
    for idx in indices:
        section_points = get_cross_section_points(mesh, spline_points, [idx])
        if section_points and section_points[0] is not None:
            section_points = section_points[0]
            section_points_list.append(section_points)
            centroid = section_points.mean(axis=0)
            section_centroids.append(centroid)
    centroids = np.array(section_centroids)

    # Insert anchors at start, side, end (top, side, bottom)
    anchors = [centre_top, side_section_centre, centre_bottom]
    # Find closest centroid to side anchor
    side_idx = np.argmin(np.linalg.norm(centroids - side_section_centre, axis=1))
    # Build ordered points: top anchor, centroids up to side, side anchor, centroids after side, bottom anchor
    ordered_points = [centre_top]
    if side_idx > 0:
        ordered_points.extend(centroids[1:side_idx])
    ordered_points.append(side_section_centre)
    if side_idx < len(centroids) - 1:
        ordered_points.extend(centroids[side_idx+1:-1])
    ordered_points.append(centre_bottom)
    ordered_points = np.array(ordered_points)

    # Interpolate a smooth, non-periodic spline through these points
    tck, u = splprep(ordered_points.T, s=0, per=0)
    u_fine = np.linspace(0, 1, num_sections)
    smoothed = np.array(splev(u_fine, tck)).T

    # Polyline trace (unsmoothed, just anchor+centroid points)
    polyline_trace = go.Scatter3d(
        x=ordered_points[:,0], y=ordered_points[:,1], z=ordered_points[:,2],
        mode='lines+markers', line=dict(color='orange', width=4), name='Centroid Polyline + Anchors (3)'
    )
    # Smoothed spline trace
    smoothed_trace = go.Scatter3d(
        x=smoothed[:,0], y=smoothed[:,1], z=smoothed[:,2],
        mode='lines+markers', line=dict(color='red', width=6), name='Smoothed Centreline (3 anchors)'
    )
    # Anchor trace
    anchor_trace = go.Scatter3d(
        x=[a[0] for a in anchors], y=[a[1] for a in anchors], z=[a[2] for a in anchors],
        mode='markers+text', marker=dict(size=8, color='black'), name='Anchors (3)',
        text=['Top','Side','Bottom'], textposition='top center'
    )
    return polyline_trace, smoothed_trace, anchor_trace, smoothed

def get_left_right_midsections(section_points, midpoint, local_axes):

    # The left-right axis is the second vector in local_axes (from get_midpoint_cross_section_from_centres)
    left_right_vec = local_axes[1]

    # Project section points onto the left-right axis (relative to the midpoint)
    relative_points = section_points - midpoint
    side_values = np.dot(relative_points, left_right_vec)

    left_section = section_points[side_values < 0]
    right_section = section_points[side_values >= 0]

    left_section_centre = left_section.mean(axis=0)
    right_section_centre = right_section.mean(axis=0)

    # Create our left and right traces
    left_midsection_trace = go.Scatter3d(
        x=left_section[:, 0], y=left_section[:, 1], z=left_section[:, 2],
        mode='markers', marker=dict(size=5, color='orange'), name='Left Midsection'
    )

    right_midsection_trace = go.Scatter3d(
        x=right_section[:, 0], y=right_section[:, 1], z=right_section[:, 2],
        mode='markers', marker=dict(size=5, color='blue'), name='Right Midsection'
    )

    left_section_centre_trace = go.Scatter3d(
        x=[left_section_centre[0]], y=[left_section_centre[1]], z=[left_section_centre[2]],
        mode='markers', marker=dict(size=8, color='orange'), name='Left Section Centre'
    )

    right_section_centre_trace = go.Scatter3d(
        x=[right_section_centre[0]], y=[right_section_centre[1]], z=[right_section_centre[2]],
        mode='markers', marker=dict(size=8, color='blue'), name='Right Section Centre'
    )
    return left_section, right_section, left_section_centre, right_section_centre,left_midsection_trace, right_midsection_trace, left_section_centre_trace, right_section_centre_trace

def get_centreline_estimate_and_split(top_circle_centre, bottom_circle_centre,
                                      left_section_centre, right_section_centre,
                                      n_points=200):
    """
    Build a closed spline centreline and split it into left and right guard cell halves
    based on top and bottom anchors.
    """
    import numpy as np
    from scipy.interpolate import CubicSpline
    import plotly.graph_objects as go

    # Full loop
    points = np.vstack([
        top_circle_centre,
        left_section_centre,
        bottom_circle_centre,
        right_section_centre,
        top_circle_centre
    ])  # repeat to close

    t = np.linspace(0, 1, len(points))
    t_fine = np.linspace(0, 1, n_points)

    spline_x = CubicSpline(t, points[:, 0], bc_type='periodic')(t_fine)
    spline_y = CubicSpline(t, points[:, 1], bc_type='periodic')(t_fine)
    spline_z = CubicSpline(t, points[:, 2], bc_type='periodic')(t_fine)
    spline = np.column_stack([spline_x, spline_y, spline_z])

    # Find closest indices to anchors
    top_idx = np.argmin(np.linalg.norm(spline - top_circle_centre, axis=1))
    bottom_idx = np.argmin(np.linalg.norm(spline - bottom_circle_centre, axis=1))

    if top_idx < bottom_idx:
        left_half = spline[top_idx:bottom_idx+1]
        right_half = np.vstack([spline[bottom_idx:], spline[:top_idx+1]])
    else:
        right_half = spline[bottom_idx:top_idx+1]
        left_half = np.vstack([spline[top_idx:], spline[:bottom_idx+1]])

    # Plot traces
    full_trace = go.Scatter3d(
        x=spline_x, y=spline_y, z=spline_z,
        mode='lines', line=dict(color='black', width=4),
        name='Full Centreline'
    )
    left_trace = go.Scatter3d(
        x=left_half[:, 0], y=left_half[:, 1], z=left_half[:, 2],
        mode='lines', line=dict(color='red', width=6),
        name='Left Guard Cell Centreline'
    )
    right_trace = go.Scatter3d(
        x=right_half[:, 0], y=right_half[:, 1], z=right_half[:, 2],
        mode='lines', line=dict(color='blue', width=6),
        name='Right Guard Cell Centreline'
    )

    return full_trace, left_trace, right_trace, left_half, right_half

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

def assign_spline_points_by_plane(
    spline_points, gc1_section_centre, gc2_section_centre, midpoint, wall_tolerance=1e-4
):
    """
    Assign each spline point to gc1, gc2, or wall using the dividing plane
    halfway between gc1 and gc2 cross-section centres.

    Parameters
    ----------
    spline_points : (N, 3) ndarray
        Points along the spline.
    gc1_section_centre : (3,) ndarray
        Centre of gc1 cross section.
    gc2_section_centre : (3,) ndarray
        Centre of gc2 cross section.
    midpoint : (3,) ndarray
        Midpoint between the two walls (same as used for cross-section plane).
    wall_tolerance : float
        Distance from midplane within which points are considered "wall".

    Returns
    -------
    spline_points_gc1, spline_points_gc2, spline_points_wall : ndarrays
    """
    # Vector from gc1 to gc2
    gc_vec = gc2_section_centre - gc1_section_centre
    gc_vec /= np.linalg.norm(gc_vec)

    # Signed distances of spline points from midplane
    signed_dist = (spline_points - midpoint) @ gc_vec

    # Assign based on which side of the midplane each point lies
    gc1_mask = signed_dist < -wall_tolerance
    gc2_mask = signed_dist > wall_tolerance
    wall_mask = np.abs(signed_dist) <= wall_tolerance

    spline_points_gc1 = spline_points[gc1_mask]
    spline_points_gc2 = spline_points[gc2_mask]
    spline_points_wall = spline_points[wall_mask]

    return spline_points_gc1, spline_points_gc2, spline_points_wall

def analyze_stomata_mesh(mesh_path, num_sections=20, n_points=40, visualize=False, dot_thresh = 0.2):
    ## Load in the mesh
    mesh = trimesh.load(mesh_path, process=False)
    ## Get the wall vertices
    wall_vertices = find_wall_vertices_vertex_normals(mesh, dot_thresh=dot_thresh)
    wall_coords = mesh.vertices[wall_vertices]

    centre_top, centre_bottom, top_wall_coords, bottom_wall_coords, top_wall_trace, bottom_wall_trace, centre_top_trace, centre_bottom_trace = get_top_bottom_wall_centres(mesh, wall_vertices)

    midpoint, traces, section_points, local_axes = get_midpoint_cross_section_from_centres(mesh, centre_top, centre_bottom)

    ## Label the left and right cross sections

    left_section, right_section, left_section_centre, right_section_centre, left_midsection_trace, right_midsection_trace, left_section_centre_trace, right_section_centre_trace = get_left_right_midsections(section_points, midpoint, local_axes)

    ## The radius of the circles to place at the top and bottom walls is taken from the radius of the midsections
    area1 = cross_section_area_2d(left_section)
    area2 = cross_section_area_2d(right_section)

    avg_area = 0.5 * (area1 + area2)

        ## Calculate the area of the circles at either end, based on the midsection areas
    radius = np.sqrt(max(avg_area, 0.0) / np.pi)

    ## Create the circles positioned at mesh ends, extending inward
    # Direction from wall toward midpoint (inward into mesh)
    top_to_mid = midpoint - centre_top
    bottom_to_mid = midpoint - centre_bottom
    
    circle_top = make_circle(top_wall_coords, radius=radius, main_axis=top_to_mid, offset_inward=True)
    circle_bottom = make_circle(bottom_wall_coords, radius=radius, main_axis=bottom_to_mid, offset_inward=True)

    circle_top_trace = get_circle_trace(circle_top, colour = 'red', name = 'Top Circle')
    circle_bottom_trace = get_circle_trace(circle_bottom, colour = 'blue', name = 'Bottom Circle')

    # Calculate circle centers for centreline anchors
    circle_top_centre = circle_top.mean(axis=0)
    circle_bottom_centre = circle_bottom.mean(axis=0)

    spline_trace, spline_x, spline_y, spline_z = get_centreline_estimate(circle_top_centre, circle_bottom_centre, left_section_centre, right_section_centre)

    ## Visualise the first step
    if visualize:
        output = visualize_mesh(mesh,[top_wall_trace, bottom_wall_trace, centre_top_trace, centre_bottom_trace, *traces, left_midsection_trace, right_midsection_trace, left_section_centre_trace, right_section_centre_trace, circle_top_trace, circle_bottom_trace, spline_trace])

    ## To simplify analysis, we will split the mesh into left and right guard cells
    left, right = split_mesh_at_wall_vertices(mesh, wall_vertices, left_section_centre, right_section_centre)
    
    # Use circle centers (not wall centers) for centreline
    full_trace, left_trace, right_trace, left_half, right_half = get_centreline_estimate_and_split(circle_top_centre, circle_bottom_centre, left_section_centre, right_section_centre, n_points=40)

    section_points_left, section_traces_left, section_bary_data_left = get_regularly_spaced_cross_sections_batch(left, left_half, circle_top_centre, circle_bottom_centre, num_sections=20)
    section_points_right, section_traces_right, section_bary_data_right = get_regularly_spaced_cross_sections_batch(right, right_half, circle_top_centre, circle_bottom_centre, num_sections=20)
    return section_points_right, section_points_left, section_traces_left, section_traces_right, [spline_x, spline_y, spline_z]
