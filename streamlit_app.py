try:
    import streamlit as st
    import tempfile
    import os
    import zipfile
    import io
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Import the stomata analysis functions
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    import mesh_io
    import wall_detection
    import cross_sections
    
    from mesh_io import load_mesh
    from wall_detection import get_wall_vertices
    from cross_sections import (
        get_top_bottom_wall_centres, get_midpoint_cross_section, make_circle, 
        cross_section_area_2d, extract_open_spline_between_anchors, 
        get_regularly_spaced_cross_sections, get_centreline_estimate
    )
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"Missing required packages. Please install: {e}")
    IMPORTS_SUCCESS = False
    # Create a dummy streamlit object for graceful degradation
    class DummySt:
        def error(self, msg): print(f"Error: {msg}")
    st = DummySt()

st.set_page_config(
    page_title="Stomata Analysis Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    if not IMPORTS_SUCCESS:
        return
        
    st.title("🔬 Stomata Analysis Tool")
    st.markdown("""
    This interactive tool allows you to analyze stomatal guard cell meshes. Upload an OBJ file to:
    - Visualize 3D mesh structure
    - Detect wall vertices and boundaries
    - Extract cross-sections and calculate areas
    - Generate centreline splines
    - Download analysis data
    """)

    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("📁 File Upload")
        uploaded_file = st.file_uploader(
            "Choose an OBJ file", 
            type=['obj'],
            help="Upload a 3D mesh file in OBJ format containing stomatal guard cells"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            st.header("⚙️ Analysis Settings")
            num_sections = st.slider(
                "Number of cross-sections", 
                min_value=5, max_value=50, value=20,
                help="Number of regular cross-sections to extract along the centreline"
            )
            
            mesh_opacity = st.slider(
                "Mesh opacity", 
                min_value=0.1, max_value=1.0, value=0.65, step=0.05,
                help="Transparency level for 3D mesh visualization"
            )
            
            show_advanced = st.checkbox("Show advanced options", value=False)
            
            if show_advanced:
                st.subheader("Advanced Settings")
                circle_radius_factor = st.slider(
                    "Circle radius factor", 
                    min_value=0.5, max_value=2.0, value=1.0, step=0.1,
                    help="Multiplier for automatically calculated circle radius"
                )
                wall_threshold = st.number_input(
                    "Wall detection threshold", 
                    min_value=1e-6, max_value=1e-3, value=1e-5, format="%.1e",
                    help="Distance threshold for detecting shared wall vertices"
                )

    # Main content area
    if uploaded_file is not None:
        # Process the uploaded file
        with st.spinner("Loading and processing mesh..."):
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.obj') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                mesh_path = tmp_file.name
            
            try:
                # Load and analyze the mesh
                mesh = load_mesh(mesh_path)
                analysis_results = analyze_mesh_detailed(mesh, num_sections)
                
                # Clean up temporary file
                os.unlink(mesh_path)
                
                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs([
                    "📊 Overview", "🎯 Mesh Analysis", " Download Results"
                ])
                
                with tab1:
                    show_overview(mesh, analysis_results)
                
                with tab2:
                    show_mesh_analysis(mesh, analysis_results, mesh_opacity)
                
                with tab3:
                    show_download_results(analysis_results, uploaded_file.name)
                
            except Exception as e:
                st.error(f"Error processing mesh: {str(e)}")
                st.info("Please ensure your OBJ file contains a valid 3D mesh with two connected components (guard cells).")
    
    else:
        # Landing page content
        st.markdown("### 🚀 Getting Started")
        st.markdown("""
        1. **Upload** an OBJ file containing stomatal guard cell mesh data
        2. **Adjust** analysis parameters in the sidebar
        3. **Explore** the different visualization tabs
        4. **Download** your analysis results
        """)
        
        st.markdown("### 📋 What this tool does:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Mesh Processing:**
            - Loads 3D mesh data
            - Splits into guard cell components
            - Detects shared wall vertices
            - Calculates mesh properties
            """)
        
        with col2:
            st.markdown("""
            **Analysis Features:**
            - Cross-section extraction
            - Area calculations
            - Centreline spline generation
            - 3D visualization
            """)
        
        # Sample data info
        st.markdown("### 📁 Sample Data Format")
        st.info("Upload OBJ files containing 3D mesh data of stomatal guard cells. The mesh should contain two connected components representing the guard cell pair.")

def analyze_mesh_detailed(mesh, num_sections=20):
    """
    Enhanced analysis function that returns comprehensive results
    """
    results = {}
    
    # Basic mesh properties
    results['mesh'] = mesh
    results['num_vertices'] = len(mesh.vertices)
    results['num_faces'] = len(mesh.faces)
    results['mesh_volume'] = mesh.volume
    results['mesh_area'] = mesh.area
    results['bounding_box'] = mesh.bounds
    
    # Split mesh into guard cells
    parts = mesh.split(only_watertight=False)
    if len(parts) < 2:
        raise ValueError("Mesh must contain at least two connected components (guard cells)")
    
    gc1, gc2 = parts[0], parts[1]
    results['gc1'] = gc1
    results['gc2'] = gc2
    results['gc1_volume'] = gc1.volume
    results['gc2_volume'] = gc2.volume
    
    # Wall detection
    gc2_wall_vertices, gc1_wall_vertices = get_wall_vertices(gc1, gc2)
    wall_coords_gc2 = gc2.vertices[gc2_wall_vertices]
    wall_coords_gc1 = gc1.vertices[gc1_wall_vertices]
    wall_coords = np.vstack([wall_coords_gc2, wall_coords_gc1])
    
    results['wall_coords'] = wall_coords
    results['gc1_wall_vertices'] = gc1_wall_vertices
    results['gc2_wall_vertices'] = gc2_wall_vertices
    results['num_wall_vertices'] = len(wall_coords)
    
    # Top/bottom wall analysis
    (centre_top, centre_bottom, top_wall_coords, bottom_wall_coords, 
     top_wall_trace, bottom_wall_trace, centre_top_trace, centre_bottom_trace) = get_top_bottom_wall_centres(mesh, wall_coords)
    
    results['centre_top'] = centre_top
    results['centre_bottom'] = centre_bottom
    results['top_wall_coords'] = top_wall_coords
    results['bottom_wall_coords'] = bottom_wall_coords
    results['wall_distance'] = np.linalg.norm(centre_top - centre_bottom)
    
    # Mid-point cross-sections
    midpoint, gc1_section_points = get_midpoint_cross_section(gc1, centre_top, centre_bottom)
    _, gc2_section_points = get_midpoint_cross_section(gc2, centre_top, centre_bottom)
    
    results['midpoint'] = midpoint
    results['gc1_section_points'] = gc1_section_points
    results['gc2_section_points'] = gc2_section_points
    
    # Area calculations
    if gc1_section_points is not None and gc2_section_points is not None:
        area1 = cross_section_area_2d(gc1_section_points)
        area2 = cross_section_area_2d(gc2_section_points)
        avg_area = 0.5 * (area1 + area2)
        radius = np.sqrt(max(avg_area, 0.0) / np.pi)
        
        results['gc1_area'] = area1
        results['gc2_area'] = area2
        results['avg_area'] = avg_area
        results['radius'] = radius
        
        # Generate circles
        top_circle = make_circle(top_wall_coords, radius)
        bottom_circle = make_circle(bottom_wall_coords, radius)
        
        results['top_circle'] = top_circle
        results['bottom_circle'] = bottom_circle
        
        circle_top_centre = top_circle.mean(axis=0)
        circle_bottom_centre = bottom_circle.mean(axis=0)
        
        gc1_section_centre = gc1_section_points.mean(axis=0)
        gc2_section_centre = gc2_section_points.mean(axis=0)
        
        results['circle_top_centre'] = circle_top_centre
        results['circle_bottom_centre'] = circle_bottom_centre
        results['gc1_section_centre'] = gc1_section_centre
        results['gc2_section_centre'] = gc2_section_centre
        
        # Centreline spline
        spline_trace, spline_x, spline_y, spline_z = get_centreline_estimate(
            circle_top_centre, circle_bottom_centre, gc1_section_centre, gc2_section_centre
        )
        
        spline = np.vstack([spline_x, spline_y, spline_z]).T
        spline_points_gc1, spline_points_gc2 = extract_open_spline_between_anchors(
            spline, circle_top_centre, circle_bottom_centre
        )
        
        results['spline'] = spline
        results['spline_points_gc1'] = spline_points_gc1
        results['spline_points_gc2'] = spline_points_gc2
        
        # Regular cross-sections
        (section_points_gc1, section_traces_gc1, section_bary_data_gc1) = get_regularly_spaced_cross_sections(
            gc1, spline_points_gc1, circle_top_centre, circle_bottom_centre, num_sections=num_sections
        )
        (section_points_gc2, section_traces_gc2, section_bary_data_gc2) = get_regularly_spaced_cross_sections(
            gc2, spline_points_gc2, circle_top_centre, circle_bottom_centre, num_sections=num_sections
        )
        
        results['section_points_gc1'] = section_points_gc1
        results['section_points_gc2'] = section_points_gc2
        results['section_traces_gc1'] = section_traces_gc1
        results['section_traces_gc2'] = section_traces_gc2
        
        # Calculate areas for each cross-section
        areas_gc1 = []
        areas_gc2 = []
        
        for section in section_points_gc1:
            if section is not None and len(section) > 3:
                areas_gc1.append(cross_section_area_2d(section))
            else:
                areas_gc1.append(0)
        
        for section in section_points_gc2:
            if section is not None and len(section) > 3:
                areas_gc2.append(cross_section_area_2d(section))
            else:
                areas_gc2.append(0)
        
        results['areas_gc1'] = areas_gc1
        results['areas_gc2'] = areas_gc2
    
    return results

def show_overview(mesh, results):
    """Display overview information about the mesh and analysis"""
    st.header("📊 Mesh Overview")
    
    # Mesh statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Vertices", f"{results['num_vertices']:,}")
    with col2:
        st.metric("Faces", f"{results['num_faces']:,}")
    with col3:
        st.metric("Volume", f"{results['mesh_volume']:.4f}")
    with col4:
        st.metric("Surface Area", f"{results['mesh_area']:.4f}")
    
    # Guard cell information
    st.subheader("Guard Cell Components")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("GC1 Volume", f"{results['gc1_volume']:.4f}")
    with col2:
        st.metric("GC2 Volume", f"{results['gc2_volume']:.4f}")
    with col3:
        st.metric("Wall Vertices", results['num_wall_vertices'])
    
    # Dimensional information
    if 'wall_distance' in results:
        st.subheader("Dimensional Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Wall Distance", f"{results['wall_distance']:.4f}")
        with col2:
            if 'avg_area' in results:
                st.metric("Average Cross-Section Area", f"{results['avg_area']:.4f}")
        with col3:
            if 'radius' in results:
                st.metric("Equivalent Circle Radius", f"{results['radius']:.4f}")

def show_mesh_analysis(mesh, results, opacity=0.65):
    """Display detailed mesh analysis with interactive 3D visualization"""
    st.header("🎯 Mesh Analysis & Visualization")
    
    # Visualization controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Visualization Controls")
        show_wall_vertices = st.checkbox("Show wall vertices", value=False)
        show_centreline = st.checkbox("Show centreline spline", value=False)
        show_circles = st.checkbox("Show top/bottom circles", value=False)
        show_cross_sections = st.checkbox("Show cross sections", value=True)
        show_wall_centres = st.checkbox("Show wall centres", value=False)
        show_tip_midsection = st.checkbox("Show tip/mid cross-sections", value=True)
    
    with col2:
        st.subheader("Orientation")
        flip_180 = st.checkbox("Flip stomata 180°", value=False, 
                              help="Rotates around Y-axis by negating X and Z coordinates (like flipping a donut over)")
        opacity = st.slider("Mesh opacity", 0.1, 1.0, 0.65, 0.05)
        
    with col3:
        st.subheader("Export Options")
        
        # Generate HTML data immediately for download
        fig_for_download = create_detailed_mesh_plot(results, opacity, show_wall_vertices, show_centreline, show_circles, show_cross_sections, flip_180, show_wall_centres, show_tip_midsection)
        html_data = fig_for_download.to_html(include_plotlyjs=True)
        
        st.download_button(
            label="📄 Download as HTML",
            data=html_data,
            file_name="stomata_analysis.html",
            mime="text/html",
            help="Download the interactive 3D plot as a standalone HTML file"
        )
    
    # Create and display the main 3D visualization
    fig = create_detailed_mesh_plot(results, opacity, show_wall_vertices, show_centreline, show_circles, show_cross_sections, flip_180, show_wall_centres)
    st.plotly_chart(fig, use_container_width=True)
    
    # Component Analysis
    st.subheader("Component Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Guard Cell 1:** {len(results['gc1'].vertices):,} vertices, {len(results['gc1'].faces):,} faces")
    with col2:
        st.write(f"**Guard Cell 2:** {len(results['gc2'].vertices):,} vertices, {len(results['gc2'].faces):,} faces")
    with col3:
        st.write(f"**Shared wall vertices:** {results['num_wall_vertices']}")

def create_detailed_mesh_plot(results, opacity=0.65, show_wall_vertices=True, show_centreline=True, show_circles=False, show_cross_sections=False, flip_180=False, show_wall_centres=False, show_tip_midsection=False):
    """Create a comprehensive 3D plot with all analysis components"""
    traces = []
    
    # Main mesh traces
    mesh = results['mesh']
    
    # Get mesh coordinates
    x_coords = mesh.vertices[:, 0]
    y_coords = mesh.vertices[:, 1] 
    z_coords = mesh.vertices[:, 2]
    
    # Apply 180-degree flip if requested (rotate around Y-axis)
    if flip_180:
        x_coords = -x_coords
        z_coords = -z_coords
    
    traces.append(go.Mesh3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color="#0072B2",
        opacity=opacity,
        name='Complete Mesh'
    ))
    
    # Wall vertices
    if show_wall_vertices and 'wall_coords' in results:
        wall_coords = results['wall_coords']
        wall_x = wall_coords[:, 0]
        wall_y = wall_coords[:, 1]
        wall_z = wall_coords[:, 2]
        
        if flip_180:
            wall_x = -wall_x
            wall_z = -wall_z
            
        traces.append(go.Scatter3d(
            x=wall_x,
            y=wall_y,
            z=wall_z,
            mode='markers',
            marker=dict(size=3, color='red'),
            name='Wall Vertices'
        ))
    
    # Top and bottom centres
    if show_wall_centres and 'centre_top' in results:
        centre_top = results['centre_top']
        centre_bottom = results['centre_bottom']
        
        top_x, top_y, top_z = centre_top[0], centre_top[1], centre_top[2]
        bottom_x, bottom_y, bottom_z = centre_bottom[0], centre_bottom[1], centre_bottom[2]
        
        if flip_180:
            top_x, top_z = -top_x, -top_z
            bottom_x, bottom_z = -bottom_x, -bottom_z
        
        traces.append(go.Scatter3d(
            x=[top_x],
            y=[top_y],
            z=[top_z],
            mode='markers',
            marker=dict(size=8, color='red', symbol='diamond'),
            name='Top Wall Centre'
        ))
        
        traces.append(go.Scatter3d(
            x=[bottom_x],
            y=[bottom_y],
            z=[bottom_z],
            mode='markers',
            marker=dict(size=8, color='blue', symbol='diamond'),
            name='Bottom Wall Centre'
        ))
    
    # Centreline spline
    if show_centreline and 'spline' in results:
        spline = results['spline']
        spline_x = spline[:, 0]
        spline_y = spline[:, 1]
        spline_z = spline[:, 2]
        
        if flip_180:
            spline_x = -spline_x
            spline_z = -spline_z
            
        traces.append(go.Scatter3d(
            x=spline_x,
            y=spline_y,
            z=spline_z,
            mode='lines',
            line=dict(width=6, color='black'),
            name='Centreline Spline'
        ))
    
    # Top and bottom circles
    if show_circles and 'top_circle' in results:
        top_circle = results['top_circle']
        bottom_circle = results['bottom_circle']
        
        top_circle_x = top_circle[:, 0]
        top_circle_y = top_circle[:, 1] 
        top_circle_z = top_circle[:, 2]
        
        bottom_circle_x = bottom_circle[:, 0]
        bottom_circle_y = bottom_circle[:, 1]
        bottom_circle_z = bottom_circle[:, 2]
        
        if flip_180:
            top_circle_x, top_circle_z = -top_circle_x, -top_circle_z
            bottom_circle_x, bottom_circle_z = -bottom_circle_x, -bottom_circle_z
            
        traces.append(go.Scatter3d(
            x=top_circle_x,
            y=top_circle_y,
            z=top_circle_z,
            mode='lines',
            line=dict(width=4, color='red'),
            name='Top Circle'
        ))
        
        traces.append(go.Scatter3d(
            x=bottom_circle_x,
            y=bottom_circle_y,
            z=bottom_circle_z,
            mode='lines',
            line=dict(width=4, color='blue'),
            name='Bottom Circle'
        ))
    
    # Cross-sections
    if show_cross_sections and 'section_points_gc1' in results and 'section_points_gc2' in results:
        # Guard Cell 1 cross-sections
        for i, section in enumerate(results['section_points_gc1']):
            if section is not None and len(section) > 0:
                section_x = section[:, 0]
                section_y = section[:, 1]
                section_z = section[:, 2]
                
                if flip_180:
                    section_x = -section_x
                    section_z = -section_z
                
                traces.append(go.Scatter3d(
                    x=section_x,
                    y=section_y,
                    z=section_z,
                    mode='lines',
                    line=dict(width=3, color='lightgray'),
                    name='GC1 Cross-sections' if i == 0 else None,
                    legendgroup='gc1_sections',
                    showlegend=(i == 0)
                ))
        
        # Guard Cell 2 cross-sections  
        for i, section in enumerate(results['section_points_gc2']):
            if section is not None and len(section) > 0:
                section_x = section[:, 0]
                section_y = section[:, 1]
                section_z = section[:, 2]
                
                if flip_180:
                    section_x = -section_x
                    section_z = -section_z
                
                traces.append(go.Scatter3d(
                    x=section_x,
                    y=section_y,
                    z=section_z,
                    mode='lines',
                    line=dict(width=3, color='lightgray'),
                    name='GC2 Cross-sections' if i == 0 else None,
                    legendgroup='gc2_sections',
                    showlegend=(i == 0)
                ))

    # Show tip/mid cross-sections
    if show_tip_midsection:
        tip = results['gc1_section_points'][0]
        mid = results['gc1_section_points'][len(results['gc1_section_points']) // 2]
        if tip is not None and len(tip) > 0:
            tip_x = tip[:, 0]
            tip_y = tip[:, 1]
            tip_z = tip[:, 2]
            if flip_180:
                tip_x = -tip_x
                tip_z = -tip_z
            traces.append(go.Scatter3d(
                x=tip_x,
                y=tip_y,
                z=tip_z,
                mode='lines',
                line=dict(width=4, color='yellow'),
                name='Tip Cross-Section'
            ))

        if mid is not None and len(mid) > 0:
            mid_x = mid[:, 0]
            mid_y = mid[:, 1]
            mid_z = mid[:, 2]
            if flip_180:
                mid_x = -mid_x
                mid_z = -mid_z
            traces.append(go.Scatter3d(
                x=mid_x,
                y=mid_y,
                z=mid_z,
                mode='lines',
                line=dict(width=4, color='magenta'),
                name='Mid Cross-Section'
            ))
        
    
    # Create figure
    fig = go.Figure(data=traces)
    
    title = "3D Mesh Analysis"
    if flip_180:
        title += " (Flipped 180°)"
        
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title=title,
        height=600
    )
    
    return fig

def show_download_results(results, original_filename="stomata"):
    """Display download options for analysis results"""
    st.header("� Download Analysis Results")
    
    # Download options in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 CSV Data")
        if st.button("Download CSV Data"):
            csv_data = create_csv_data(results)
            st.download_button(
                label="Click to Download",
                data=csv_data,
                file_name="stomata_analysis.csv",
                mime="text/csv"
            )
    
    with col2:
        st.subheader("🖼️ Cross-Section Images")
        if st.button("Download Cross-Section PNGs"):
            if 'section_points_gc1' in results and 'section_points_gc2' in results:
                with st.spinner("Generating PNG images..."):
                    zip_data = create_cross_section_pngs_package(results, original_filename)
                    st.download_button(
                        label="Click to Download ZIP",
                        data=zip_data,
                        file_name="cross_sections.zip",
                        mime="application/zip"
                    )
            else:
                st.warning("No cross-section data available for PNG generation.")
    
    # Data preview
    st.subheader("Data Preview")
    
    # Create preview dataframe
    preview_data = {}
    if 'areas_gc1' in results:
        preview_data['GC1_Areas'] = results['areas_gc1']
    if 'areas_gc2' in results:
        preview_data['GC2_Areas'] = results['areas_gc2']
    
    if preview_data:
        preview_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in preview_data.items()]))
        st.dataframe(preview_df.head(10), hide_index=True)
        
        if len(preview_df) > 10:
            st.info(f"Showing first 10 rows. Complete dataset contains {len(preview_df)} rows.")

def create_cross_section_png(section_points, section_name, dpi=150, size_inches=(4, 4)):
    """
    Generate a 2D PNG image of a cross-section with black lines on white background.
    
    Parameters:
    -----------
    section_points : numpy.ndarray
        3D coordinates of the cross-section points
    section_name : str
        Name for the cross-section (used in title)
    dpi : int
        Resolution of the output image
    size_inches : tuple
        Size of the figure in inches (width, height)
    
    Returns:
    --------
    bytes
        PNG image data as bytes
    """
    if section_points is None or len(section_points) < 3:
        return None
    
    # Project 3D points to 2D using PCA
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(section_points)
    
    # Create figure with white background
    fig, ax = plt.subplots(1, 1, figsize=size_inches, facecolor='white')
    ax.set_facecolor('white')
    
    # Close the contour by adding the first point at the end
    closed_points = np.vstack([points_2d, points_2d[0]])
    
    # Plot the cross-section as a black line
    ax.plot(closed_points[:, 0], closed_points[:, 1], 'k-', linewidth=2)
    
    # Set equal aspect ratio and clean up axes
    ax.set_aspect('equal')
    ax.set_title(section_name, fontsize=12, pad=10)
    
    # Remove axes ticks and labels for clean appearance
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add a thin border
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.5)
    
    # Tight layout to minimize white space
    plt.tight_layout()
    
    # Save to bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)  # Important: close figure to free memory
    
    img_buffer.seek(0)
    return img_buffer.getvalue()

def create_cross_section_pngs_package(results, original_filename):
    """
    Create a ZIP package containing PNG images of all cross-sections.
    
    Parameters:
    -----------
    results : dict
        Analysis results containing section_points_gc1 and section_points_gc2
    original_filename : str
        Original filename for naming convention (e.g., "stomateID_timepoint.obj")
        
    Returns:
    --------
    bytes
        ZIP file data as bytes
    """
    zip_buffer = io.BytesIO()
    
    # Extract base name without extension for better naming
    base_name = os.path.splitext(original_filename)[0]
    
    # Try to parse stomata ID and timepoint from filename
    # Common patterns: "stomateID_timepoint" or "ID_timepoint_other" 
    name_parts = base_name.replace('-', '_').split('_')
    
    if len(name_parts) >= 2:
        stomata_id = name_parts[0]
        timepoint = name_parts[1]
        name_prefix = f"{stomata_id}_{timepoint}"
    else:
        # Fallback to original base name if pattern doesn't match
        name_prefix = base_name
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        
        # Process GC1 cross-sections
        if 'section_points_gc1' in results:
            for i, section in enumerate(results['section_points_gc1']):
                if section is not None and len(section) > 2:
                    section_name = f"GC1 - Section {i:02d}"
                    png_data = create_cross_section_png(section, section_name)
                    if png_data:
                        filename = f"{name_prefix}_GC1_section_{i:02d}.png"
                        zip_file.writestr(filename, png_data)
        
        # Process GC2 cross-sections
        if 'section_points_gc2' in results:
            for i, section in enumerate(results['section_points_gc2']):
                if section is not None and len(section) > 2:
                    section_name = f"GC2 - Section {i:02d}"
                    png_data = create_cross_section_png(section, section_name)
                    if png_data:
                        filename = f"{name_prefix}_GC2_section_{i:02d}.png"
                        zip_file.writestr(filename, png_data)
        
        # Add a README file
        readme_content = f"""Cross-Section PNG Images for {original_filename}
========================================

This package contains 2D PNG images of all extracted cross-sections.

File Naming Convention:
- {name_prefix}_GC1_section_XX.png: Cross-sections from Guard Cell 1
- {name_prefix}_GC2_section_XX.png: Cross-sections from Guard Cell 2

Where:
- Stomata ID and timepoint are extracted from the original filename
- GC1/GC2 indicates the guard cell (Guard Cell 1 or Guard Cell 2)  
- XX is the section number (00, 01, 02, etc.)

Each image shows the cross-section outline as a black line on a white background.
Images are generated at 150 DPI and sized at 4x4 inches for high quality.

Generated by Stomata Analysis Tool
Timestamp: {pd.Timestamp.now().isoformat()}
"""
        zip_file.writestr("README.txt", readme_content)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def create_csv_data(results):
    """Create CSV data from analysis results"""
    data = []
    
    # Cross-section data
    if 'areas_gc1' in results and 'areas_gc2' in results:
        max_len = max(len(results['areas_gc1']), len(results['areas_gc2']))
        
        for i in range(max_len):
            row = {'section_index': i}
            
            if i < len(results['areas_gc1']):
                row['gc1_area'] = results['areas_gc1'][i]
            else:
                row['gc1_area'] = None
                
            if i < len(results['areas_gc2']):
                row['gc2_area'] = results['areas_gc2'][i]
            else:
                row['gc2_area'] = None
            
            # Add coordinates if available
            if 'section_points_gc1' in results and i < len(results['section_points_gc1']):
                section = results['section_points_gc1'][i]
                if section is not None:
                    row['gc1_num_points'] = len(section)
                    row['gc1_centroid_x'] = np.mean(section[:, 0])
                    row['gc1_centroid_y'] = np.mean(section[:, 1])
                    row['gc1_centroid_z'] = np.mean(section[:, 2])
            
            if 'section_points_gc2' in results and i < len(results['section_points_gc2']):
                section = results['section_points_gc2'][i]
                if section is not None:
                    row['gc2_num_points'] = len(section)
                    row['gc2_centroid_x'] = np.mean(section[:, 0])
                    row['gc2_centroid_y'] = np.mean(section[:, 1])
                    row['gc2_centroid_z'] = np.mean(section[:, 2])
            
            data.append(row)
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

if __name__ == "__main__":
    main()
