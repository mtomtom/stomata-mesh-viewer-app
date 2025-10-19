import plotly.graph_objects as go

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

## Code for creating traces for cross-sections and splines
def create_cross_section_trace(section_points, colour, name):
    trace = go.Scatter3d(
        x=section_points[:,0],  # X-coordinate
        y=section_points[:,1],  # Y-coordinate
        z=section_points[:,2],  # Z-coordinate
        mode='lines',     # Only markers
        line=dict(width=10, color=colour),  # Marker style
        name=name,
        opacity=1         # Legend name
    )
    return trace

def create_spline_trace(spline_points, colour, name):
    trace = go.Scatter3d(
        x=spline_points[:,0],
        y=spline_points[:,1],
        z=spline_points[:,2],
        mode='lines',
        line=dict(width=6, color=colour),
        name=name
    )
    return trace

def create_centre_trace(centre_point, colour, name):
    trace = go.Scatter3d(
        x=[centre_point[0]],
        y=[centre_point[1]],
        z=[centre_point[2]],
        mode='markers',
        marker=dict(size=12, color=colour),
        name=name
    )
    return trace
