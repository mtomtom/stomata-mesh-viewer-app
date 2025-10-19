from .mesh_io import load_mesh
from .wall_detection import get_wall_vertices
from .cross_sections import get_top_bottom_wall_centres, get_midpoint_cross_section, make_circle, cross_section_area_2d, extract_open_spline_between_anchors, get_regularly_spaced_cross_sections, get_centreline_estimate
from .plotting import visualize_mesh

import numpy as np

def analyze_mesh(mesh_path):
    mesh = load_mesh(mesh_path)

    ## Split mesh into two connected components (guard cells)
    parts = mesh.split(only_watertight=False)  
    gc1, gc2 = parts[0], parts[1]

    ## Find the wall vertices
    gc2_wall_vertices, gc1_wall_vertices = get_wall_vertices(gc1, gc2)

    ## Find the wall coordinates
    wall_coords_gc2 = gc2.vertices[gc2_wall_vertices]
    wall_coords_gc1 = gc1.vertices[gc1_wall_vertices]
    wall_coords = np.vstack([wall_coords_gc2, wall_coords_gc1])

    centre_top, centre_bottom, top_wall_coords, bottom_wall_coords, top_wall_trace, bottom_wall_trace, centre_top_trace, centre_bottom_trace = get_top_bottom_wall_centres(mesh, wall_coords)
    midpoint, gc1_section_points = get_midpoint_cross_section(gc1, centre_top, centre_bottom)
    midpoint, gc2_section_points = get_midpoint_cross_section(gc2, centre_top, centre_bottom)

    # Circle radius from midsections
    area1 = cross_section_area_2d(gc1_section_points)
    area2 = cross_section_area_2d(gc2_section_points)
    avg_area = 0.5 * (area1 + area2)
    radius = np.sqrt(max(avg_area, 0.0) / np.pi)

    top_circle = make_circle(top_wall_coords, radius)
    bottom_circle = make_circle(bottom_wall_coords, radius)

    circle_top_centre = top_circle.mean(axis=0)
    circle_bottom_centre = bottom_circle.mean(axis=0)

    gc1_section_centre = gc1_section_points.mean(axis=0)
    gc2_section_centre = gc2_section_points.mean(axis=0)

    spline_trace, spline_x, spline_y, spline_z = get_centreline_estimate(circle_top_centre, circle_bottom_centre, gc1_section_centre, gc2_section_centre)

    # After creating the full closed spline:
    spline = np.vstack([spline_x, spline_y, spline_z]).T
    spline_points_gc1, spline_points_gc2 = extract_open_spline_between_anchors(spline, circle_top_centre, circle_bottom_centre)

    section_points_gc1, section_traces_gc1, section_bary_data_gc1 = get_regularly_spaced_cross_sections(gc1, spline_points_gc1, circle_top_centre, circle_bottom_centre, num_sections=20)
    section_points_gc2, section_traces_gc2, section_bary_data_gc2 = get_regularly_spaced_cross_sections(gc2, spline_points_gc2, circle_top_centre, circle_bottom_centre, num_sections=20)

    tip_gc1 = section_points_gc1[0]
    mid_index_gc1 = len(section_points_gc1) // 2
    midsection_gc1 = section_points_gc1[mid_index_gc1]
    tip_gc2 = section_points_gc2[0]
    mid_index_gc2 = len(section_points_gc2) // 2
    midsection_gc2 = section_points_gc2[mid_index_gc2]

    return tip_gc1, midsection_gc1, tip_gc2, midsection_gc2, gc1_section_centre, gc2_section_centre, spline_points_gc1, spline_points_gc2

def visualize_mesh_analysis(mesh, extra_details=None, title="Mesh Visualization"):
    return visualize_mesh(mesh, extra_details=extra_details, title=title)