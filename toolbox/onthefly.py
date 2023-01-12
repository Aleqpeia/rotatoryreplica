import sys
from datetime import datetime

import numpy as np
from scipy.linalg import eigh

pull_topology = str(sys.argv[1])
pull_trajectory = str(sys.argv[2])

trajectory_file = pull_trajectory + '.xtc'
topology_file = pull_topology + '.gro'

def output_structure_name(type=None, extension=".gro", hms=False):

    # Get the formatted date
    if hms:
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        current_date = datetime.now().strftime("%Y%m%d")

    # Get the name
    file_name = current_date
    if type is not None:
        file_name += "_" + type.upper()
    file_name += extension

    return output_structure_name

def output_trr_name(type=None, extension=".trr", hms=False):

    # Get the formatted date
    if hms:
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        current_date = datetime.now().strftime("%Y%m%d")

    # Get the name
    file_name = current_date
    if type is not None:
        file_name += "_" + type.upper()
    file_name += extension

    return output_trr_name

def output_xtc_name(type=None, extension=".trr", hms=False):

    # Get the formatted date
    if hms:
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        current_date = datetime.now().strftime("%Y%m%d")

    # Get the name
    file_name = current_date
    if type is not None:
        file_name += "_" + type.upper()
    file_name += extension

    return output_trr_name
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

u = mda.Universe(sys.argv[1])

protein = u.select_atoms('PROTEIN')


def rotatory(protein):
    # Calculate the center of geometry of the protein
    cog = protein.center_of_geometry()

    # Define the rotational angles (in degrees) for each degree of freedom
    rot_x = 15.0  # rotation around the x-axis
    rot_y = 15.0  # rotation around the y-axis
    rot_z = 45.0  # rotation around the z-axis

    # Convert the rotational angles to radians
    rot_x_rad = np.radians(rot_x)
    rot_y_rad = np.radians(rot_y)
    rot_z_rad = np.radians(rot_z)

    # Create the rotation matrix for each degree of freedom
    rot_matrix_x = np.array([[1, 0, 0],
                             [0, np.cos(rot_x_rad), -np.sin(rot_x_rad)],
                             [0, np.sin(rot_x_rad), np.cos(rot_x_rad)]])

    rot_matrix_y = np.array([[np.cos(rot_y_rad), 0, np.sin(rot_y_rad)],
                             [0, 1, 0],
                             [-np.sin(rot_y_rad), 0, np.cos(rot_y_rad)]])

    rot_matrix_z = np.array([[np.cos(rot_z_rad), -np.sin(rot_z_rad), 0],
                             [np.sin(rot_z_rad), np.cos(rot_z_rad), 0],
                             [0, 0, 1]])
    for atom in protein:
        # Shift the atom coordinates so that the center of geometry is at the origin
        atom.position -= cog

        # Apply the rotations in sequence (z-y-x order)
        atom.position = np.dot(rot_matrix_z, atom.position)
        atom.position = np.dot(rot_matrix_y, atom.position)
        atom.position = np.dot(rot_matrix_x, atom.position)

        # Shift the atom coordinates back so that the center of geometry is at the original position
        atom.position += cog

form = u.atoms - protein.atoms
rot = rotatory(protein)
output = rot + form
output.write(sys.argv[3])


def diffusion_maps_DMD(trajectory, alpha=0.5, t=int(sys.argv[3]), n_components=2):

    # Performs on-the-fly diffusion maps for directed molecular dynamics.
    #Parameters:
    #trajectory: a 2D numpy array of shape (n_timesteps, n_dims) representing a molecular dynamics trajectory
    #alpha: the decay rate of the diffusion map kernel
    #t: the time scale of the diffusion map
    #Returns:
    #eigenvectors: a 2D numpy array of shape (n_timesteps, n_eigenvectors) representing the eigenvectors of the diffusion map
    #eigenvalues: a 1D numpy array of shape (n_eigenvectors,) representing the eigenvalues of the diffusion map


    # Compute the distance matrix between all pairs of frames in the trajectory
    distances = np.zeros((trajectory.shape[0], trajectory.shape[0]))
    for i in range(trajectory.shape[0]):
        for j in range(trajectory.shape[0]):
            distances[i, j] = np.linalg.norm(trajectory[i] - trajectory[j])


    kernel = np.exp(-alpha * np.power(distances, 2) / (2 * t))     # diffusion map (Gaussian kernel)

    # Normalize the rows of the kernel
    row_sums = np.sum(kernel, axis=1)
    kernel = kernel / row_sums[:, np.newaxis]

    # Compute the eigenvectors and eigenvalues of the kernel
    eigenvalues, eigenvectors = eigh(kernel)

    # Return the eigenvectors and eigenvalues, sorted by descending eigenvalue
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sort_indices]
    eigenvalues = eigenvalues[sort_indices]

    # Normalize the eigenvectors
    eigenvectors_norm = np.apply_along_axis(normalize, 1, eigenvectors[:, :n_components])
    return eigenvectors_norm, eigenvalues, eigenvectors




