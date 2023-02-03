import os
import zipfile

import MDAnalysis as mda
import matplotlib as mpl
import numpy as np
import tqdm
from MDAnalysis.lib.mdamath import triclinic_vectors
from numpy import arange, hstack
from scipy.spatial import Voronoi

from core import _check_extension, _get_metadata_content


# Load the GROMACS trajectory and the corresponding topology file




def getTypes(u):
    molecule_types = u.atoms.resnames
    return molecule_types


def cleartraj(u):
    membrane = u.select_atoms('MEMBRANE')
    membrane.atoms.wrap(center='mass')

# Write a new trajectory containing only protein atoms and centered in the box
    with mda.Writer('centered_protein.xtc', membrane.atoms.n_atoms) as W:
        for ts in u.trajectory:
            W.write(membrane)


# # Select all protein atoms
# membrane = u.select_atoms('MEMBRANE')
#
# # Center the protein in the simulation box
# membrane.atoms.wrap(center='mass')
#
# # Write a new trajectory containing only protein atoms and centered in the box
# with mda.Writer('centered_protein.xtc', membrane.atoms.n_atoms) as W:
#     for ts in u.trajectory:
#         W.write(membrane)

def getPos(u, molecule_types):
    position_array = u.atoms.positions
    new_position_array = np.zeros((len(molecule_types), len(position_array), 3))
    for i in range(len(molecule_types)):
        new_position_array[i] = position_array
    return new_position_array

def map_molecule(selection, heavy_selection):

    # Load the information arrays from the file
    atom_name_array = selection.names
    atom_id_array = selection.atoms.ids
    nonH_id_array = heavy_selection.atoms.ids
    bonds_id_array = heavy_selection.bonds._bix

    # Build the index-name dictionaries
    atom_name_dict = {atom_id_array[k]: v for k, v in enumerate(atom_name_array)}
    nonH_id_dict = {k: v for v, k in enumerate(nonH_id_array)}

    # Construct the bonds array by name
    bonds_name_array = np.zeros(bonds_id_array.shape).astype(str)
    for old_index, new_index in atom_name_dict.items():
        bonds_name_array[bonds_id_array == old_index] = new_index

    # Clean the bond name array from hydrogens
    cleaned_bonds_id_array = []
    cleaned_bonds_names_array = []

    for i, pair in enumerate(bonds_name_array):
        a, b = pair
        if 'H' not in a and 'H' not in b:
            cleaned_bonds_id_array.append(bonds_id_array[i])
            cleaned_bonds_names_array.append(pair)

    cleaned_bonds_id_array = np.array(cleaned_bonds_id_array)
    cleaned_bonds_names_array = np.array(cleaned_bonds_names_array)

    # Construct the bond array by id
    nonH_bonds_id_array = np.copy(cleaned_bonds_id_array)

    for old_index, new_index in nonH_id_dict.items():
        nonH_bonds_id_array[cleaned_bonds_id_array == old_index] = new_index

    nonH_bonds_id_array = [(a, b) for a, b in nonH_bonds_id_array]

    # Generate the dictionaries
    heavy_atoms = {
    'names': cleaned_bonds_names_array,
    'ids': np.array(nonH_bonds_id_array),
    'old_ids': cleaned_bonds_id_array,
    }

    all_atoms = {
    'names': bonds_name_array,
    'ids': bonds_id_array,
    }

    return all_atoms, heavy_atoms
def get_type_info(u, type):


    resids = np.unique(u.select_atoms("resname "+type).resids )
    n_molecules = resids.shape[0]

    # Get the atom selections
    atom_selection = u.select_atoms("resid"+str(resids[0]))
    heavy_atom_selection = u.select_atoms("resid"+str(resids[0])+"and not type H*")

    # Get the names
    atom_names = atom_selection.names
    n_atoms = atom_names.shape[0]
    heavy_atom_names = heavy_atom_selection.names
    n_heavy_atoms = heavy_atom_names.shape[0]

    # Get the masses
    atom_masses = atom_selection.masses
    heavy_atom_masses = heavy_atom_selection.masses

    # Get the IDs of the heavy atoms
    heavy_atom_ids = np.array( [i for i, name in enumerate(atom_names) if name in heavy_atom_names] )

    # Get the bonds
    atom_bonds, heavy_atom_bonds = map_molecule(atom_selection, heavy_atom_selection)

    molecule_info = {'resids': resids, 'n_molecules': n_molecules,
                     'atoms': dict(names=atom_names, number=n_atoms, masses=atom_masses, bonds=atom_bonds),
                     'heavy_atoms': dict(names=heavy_atom_names, number=n_heavy_atoms, masses=heavy_atom_masses,
                      ids=heavy_atom_ids, bonds=heavy_atom_bonds)}

    return molecule_info

def getCOM(positions, masses):

    mass_ive = np.tile(masses, (positions.shape[0], positions.shape[1], 1))
    weighted_positions = positions * mass_ive[:,:,:, np.newaxis] # Multiply the position by the atomic weights
    center_of_masses = np.sum(weighted_positions, axis=2) / masses.sum()
    return center_of_masses

def gyration_tensor(position, mass):

    mass_array = np.tile(mass, (position.shape[0], 1))
    weighted_position = position * mass_array[:, :, np.newaxis]

    xx = np.sum(weighted_position[:,:,0]*position[:,:,0], axis=1)
    xy = np.sum(weighted_position[:,:,0]*position[:,:,1], axis=1)
    xz = np.sum(weighted_position[:,:,0]*position[:,:,2], axis=1)
    yy = np.sum(weighted_position[:,:,1]*position[:,:,1], axis=1)
    yz = np.sum(weighted_position[:,:,1]*position[:,:,2], axis=1)
    zz = np.sum(weighted_position[:,:,2]*position[:,:,2], axis=1)

    gyration_tensors = np.swapaxes( np.vstack([xx,xy,xz,xy,yy,yz,xz,yz,zz]), 0, 1)

    return np.reshape(gyration_tensors, (gyration_tensors.shape[0],3,3)) / np.sum(mass)

def cartesian2polar(positions):

    # Initialise the new set of coordinates
    polar_position = np.zeros((positions.shape[0], positions.shape[1], positions.shape[2], 2))

    # Calculate the position in the new set of coordinates
    polar_position[:,:,:, 0] = np.sqrt(positions[:,:,:, 0] ** 2 + positions[:,:,:, 1] ** 2)
    polar_position[:,:,:, 1] = positions[:,:,:, 2]

    return polar_position

def computeDistances(positions, ranked_bonds_ids):


    # Reshape the position array
    moleculeNbr = positions.shape[1]
    atomNbr = positions.shape[2]
    positions = np.reshape(positions, (positions.shape[0], moleculeNbr*atomNbr, 3))

    # Loop over all the frames
    all_distances = []
    for frame in tqdm(range(positions.shape[0]), desc='Computing distances...'):

        # Loop over all the distance pairs
        a = arange(0)
        b = arange(0)
        for (i, j) in ranked_bonds_ids:
            a = hstack((a, (i + arange(moleculeNbr) * atomNbr)))
            b = hstack((b, (j + arange(moleculeNbr) * atomNbr)))

        # Calculate the distances and return the resulting array
        vectdist = (positions[frame][a] - positions[frame][b])

        all_distances.append( (vectdist ** 2).sum(axis=1) ** 0.5 )

    # Reshape the distance array
    all_distances = np.array(all_distances)
    all_distances = np.reshape(all_distances, (all_distances.shape[0], ranked_bonds_ids.shape[0], moleculeNbr))
    all_distances = np.swapaxes(all_distances, 1,2)

    return all_distances

def get_neighbors(cells, threshold=.01):

    cell_ids = np.array([cell.id for cell in cells])

    # Loop over all cells
    neighbors_dictionnary = {}
    for cell in cells:

        neighbors = np.array( cell.neighbors() )
        face_area = np.array( cell.face_areas() )
        # Remove the current cell from the list
        current_id = cell.id
        face_area = face_area[neighbors != current_id]
        neighbors = neighbors[neighbors != current_id]

        # Apply the threshold
        area_threshold = np.sum(face_area) * threshold
        neighbors = neighbors[face_area >= area_threshold]

        neighbors_dictionnary[current_id] = np.copy(neighbors)

    return neighbors_dictionnary
def get_vertices(cells):

    # Loop over all cells
    vertices_dictionnary = {}
    for cell in cells:
        vertices_dictionnary[cell.id] = cell.vertices()

    return vertices_dictionnary
def clean_neighbours(neighbours, all_ids, accepted_ids):

    filtered_neighbours = [ x for x in neighbours if all_ids[x] in accepted_ids ]

    return filtered_neighbours



def find_leaflets(center_of_masses):

    # Find the Z position of the membrane mid-plane
    membrane_mid_plane = np.mean(center_of_masses, axis=1)[:,2]

    # Initialise the leaflets
    leaflets = np.zeros((center_of_masses.shape[0], center_of_masses.shape[1])).astype('U256')
    leaflets[:,:] = 'top'

    # Assign the leaflets
    leaflets[center_of_masses[:,:,2] < membrane_mid_plane[:,np.newaxis]] = 'bottom'

    return leaflets

def mean_LCOP(u,resname,leaflet,atoms):
    leaflet = u.select_atoms("resname {} and {}".format(resname,leaflet))
    leaflet_atoms = leaflet.select_atoms(atoms)
    normal = triclinic_vectors(leaflet.dimensions)[2]
    vec_arr = leaflet_atoms.positions - leaflet.center_of_mass()
    vec_arr = vec_arr / np.linalg.norm(vec_arr, axis=1)[:, np.newaxis]
    LCOP = np.mean(np.dot(vec_arr, normal))
    return LCOP



def velocity_parameters(universe, group_selection):

    # Select the group of atoms
    group = universe.select_atoms(group_selection)

    # Initialize a dictionary to store the velocity parameters
    velocity_parameters = {"frame":[], "mean_velocity": [], "velocity_std": []}
    mda.traj.base.VelocityWriter()

    # Loop over all frames in the trajectory
    for ts in universe.trajectory:
        for molecule in group:
            # Calculate the mean velocity and velocity standard deviation for the group of atoms
            mean_velocity = np.mean(molecule.velocities, axis=0)
            velocity_std = np.std(molecule.velocities, axis=0)

            # Append the frame number, mean velocity, and velocity standard deviation to the dictionary
            velocity_parameters["frame"].append(ts.frame)
            velocity_parameters["mean_velocity"].append(mean_velocity)
            velocity_parameters["velocity_std"].append(velocity_std)

        return velocity_parameters

def LCOP_map(u,leaflet,atoms):

    # Get a list of unique lipid residues
    resnames = u.atoms.resnames
    resnames = list(set(resnames))
    LCOP_per_lipid = {}
    normal = triclinic_vectors(u.dimensions)[2]
    for resname in resnames:
        leaflet_res = u.select_atoms("resname {} and {}".format(resname,leaflet))
        leaflet_atoms = leaflet_res.select_atoms(atoms)
        vec_arr = leaflet_atoms.positions - leaflet_res.center_of_mass()
        vec_arr = vec_arr / np.linalg.norm(vec_arr, axis=1)[:, np.newaxis]
        LCOP = np.dot(vec_arr, normal)
        LCOP_per_lipid[resname] = LCOP
    return LCOP_per_lipid

def mosaic(center_of_masses, LCOP_per_lipid, box, threshold=0.01):

    # Copy the input
    computed_COMs = np.copy(center_of_masses)
    computed_box = np.copy(box)
    computed_LCOPs = np.copy(LCOP_per_lipid)


    # Modify the dimensions
    computed_COMs[:,2] = 0.5
    computed_box[2] = 1
    # Call the 3D tessellations
    premosaic = Voronoi(computed_COMs)
    volumes, vertices, neighbours = premosaic
    minima = np.min(computed_LCOPs, axis=0)
    maxima = np.max(computed_LCOPs, axis=0)
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)


    return volumes, vertices, neighbours

def openModelFile(file_path):

    # Check the extension of the file
    _check_extension(file_path, extensions=[".lpm"], stop=True)

    # Get the file from the archive
    compressed_file = zipfile.ZipFile(file_path)

    # Process all the data files
    for file in compressed_file.namelist():

        current_data = compressed_file.read(file)

        current_file = open(file, 'w')
        current_file.write(current_data.decode("utf-8"))
        current_file.close()

    # Extract the training sets from the file
    coordinates = np.loadtxt('model_coordinates.csv', delimiter=',')
    distances = np.loadtxt('model_distances.csv', delimiter=',')
    phases = np.loadtxt('model_phases.csv', delimiter=',', dtype=str)

    # Extract the training parameters from the file
    general_info, training_infos, training_scores, training_errors = _get_metadata_content('model_data.xml')
    metadata_content = {
    'general': general_info,
    'training': training_infos,
    'scores': {
    'scores' : training_scores,
    'errors' : training_errors
    }
    }

    # Delete the files after loading informations
    for file in compressed_file.namelist():
        os.remove(file)
    compressed_file.close()

    return coordinates, distances, phases, metadata_content

def mosaic_bilayer(center_of_masses, boxes, ids, leaflets, threshold=0.01):

    # Process all the frames
    all_volumes = []
    all_vertices = []
    all_neighbours = []
    for frame in tqdm(range(center_of_masses.shape[0]), desc="Computing tessellations..."):

        # Extract the information of the current frame
        current_COMs = center_of_masses[frame]
        current_box = boxes[frame]
        current_leaflet = leaflets[frame]

        # Split the system in leaflets
        top_COMs = current_COMs[current_leaflet == 'top']
        top_ids = ids[current_leaflet == 'top']
        bottom_COMs = current_COMs[current_leaflet == 'bottom']
        bottom_ids = ids[current_leaflet == 'bottom']

        # Process each leaflet
        current_top_volumes, current_top_vertices, current_top_neighbours = 2d_tessellations(top_COMs, current_box, threshold=threshold)
        current_bottom_volumes, current_bottom_vertices, current_bottom_neighbours = 2d_tessellations(bottom_COMs, current_box, threshold=threshold)

        # Merge the leaflets
        current_volumes = np.zeros(current_leaflet.shape)
        current_volumes[current_leaflet == 'top'] = current_top_volumes
        current_volumes[current_leaflet == 'bottom'] = current_bottom_volumes
        all_volumes.append(current_volumes)

        # Save the vertices and neighbours
        current_vertices = {}
        current_neighbours = {}

        for i, id in enumerate(top_ids):
            current_vertices[id] = current_top_vertices[i]
            current_neighbours[id] = current_top_neighbours[i]

        for i, id in enumerate(bottom_ids):
            current_vertices[id] = current_bottom_vertices[i]
            current_neighbours[id] = current_bottom_neighbours[i]

        all_vertices.append(current_vertices)
        all_neighbours.append(current_neighbours)

    # Convert in array
    all_volumes = np.array(all_volumes)

    return all_volumes, all_vertices, all_neighbours