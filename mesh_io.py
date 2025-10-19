import trimesh

def load_mesh(file_path, process=False):
    """
    Load a mesh file using trimesh.

    Parameters
    ----------
    file_path : str
        Path to the mesh file (e.g., .obj, .ply, .stl).
    process : bool, optional
        Whether to process the mesh for watertightness, normals, etc. (default: False)

    Returns
    -------
    trimesh.Trimesh
        The loaded mesh object.
    """
    return trimesh.load(file_path, process=process)

def save_mesh(mesh, file_path):
    """
    Save a trimesh mesh object to a file.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh object to save.
    file_path : str
        Path to save the mesh file.
    """
    mesh.export(file_path)