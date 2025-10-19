# Stomata Analysis Tool

A Streamlit web application for analyzing stomatal guard cell meshes.

## Features

- **3D Mesh Visualization**: Interactive 3D visualization of guard cell meshes
- **Wall Detection**: Automatic detection of shared wall vertices between guard cells
- **Cross-Section Analysis**: Extract and analyze cross-sections along the centreline
- **Area Calculations**: Compute cross-sectional areas and statistics
- **Centreline Splines**: Generate smooth centreline paths through the stomatal pore
- **Data Export**: Download analysis results in CSV, JSON, and complete ZIP packages

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

## Usage

1. **Upload**: Select an OBJ file containing stomatal guard cell mesh data
2. **Configure**: Adjust analysis parameters in the sidebar
3. **Explore**: Navigate through different tabs to view:
   - Overview statistics
   - 3D mesh analysis and visualization
   - Cross-section details
   - Quantitative data analysis
   - Download options
4. **Download**: Export your analysis results

## Input Requirements

- **File Format**: OBJ files containing 3D mesh data
- **Mesh Structure**: The mesh should contain two connected components representing a pair of guard cells
- **Coordinate System**: Standard 3D coordinates (X, Y, Z)

## Analysis Pipeline

The tool performs the following analysis steps:

1. **Mesh Loading**: Load and validate the 3D mesh
2. **Component Separation**: Split mesh into individual guard cells
3. **Wall Detection**: Identify shared vertices between guard cells
4. **Boundary Analysis**: Detect top and bottom wall boundaries
5. **Cross-Section Extraction**: Generate regular cross-sections along the centreline
6. **Area Calculation**: Compute cross-sectional areas using 2D convex hull
7. **Spline Generation**: Create smooth centreline splines
8. **Data Export**: Package results for download

## Output Data

The analysis generates:

- **CSV Files**: Cross-section coordinates and area measurements
- **JSON Reports**: Comprehensive analysis statistics and metadata
- **3D Coordinates**: Wall vertices, spline points, and section coordinates
- **Visualization**: Interactive 3D plots and 2D charts

## Dependencies

- `streamlit`: Web app framework
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `plotly`: Interactive visualizations
- `trimesh`: 3D mesh processing
- `scikit-learn`: Machine learning utilities (PCA, clustering)
- `scipy`: Scientific computing (interpolation, spatial algorithms)

## File Structure

```
stomata_analysis/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── mesh_io.py               # Mesh loading/saving functions
├── wall_detection.py        # Wall vertex detection
├── cross_sections.py        # Cross-section analysis
├── plotting.py             # 3D visualization functions
└── pipeline.py             # Main analysis pipeline
```

## Troubleshooting

**Import Errors**: Ensure all packages in `requirements.txt` are installed
**Mesh Loading Issues**: Verify OBJ file contains valid 3D mesh data
**Analysis Failures**: Check that mesh has exactly two connected components

## Example Usage

```python
from streamlit_app import analyze_mesh_detailed

# Load and analyze mesh
results = analyze_mesh_detailed(mesh, num_sections=20)

# Access results
print(f"Total volume: {results['mesh_volume']}")
print(f"Number of wall vertices: {results['num_wall_vertices']}")
print(f"Cross-section areas: {results['areas_gc1']}")
```